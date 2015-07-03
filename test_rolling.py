import numpy
import theano
from simplelearn.utils import safe_izip
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.nodes import RescaleImage, FormatNode, Conv2dLayer, SoftmaxLayer, CrossEntropy, Misclassification, AffineLayer
from simplelearn.training import SavesAtMinimum, AverageMonitor, ValidationCallback, SavesAtMinimum, StopsOnStagnation, LimitsNumEpochs
from simplelearn.io import SerializableModel
from extension_semi_stochastic import SemiSgd, SemiSgdParameterUpdater
import matplotlib.pyplot as plt
import time
import pdb
from sklearn.utils import shuffle



shuffle_dataset = True

training_set, testing_set = load_mnist()

training_tensors = [t[:50000, ...] for t in training_set.tensors]  # the first 50000 examples
validation_tensors = [t[50000:, ...] for t in training_set.tensors]  # the remaining 10000 examples


if shuffle_dataset == True:
    '''
    def shuffle_in_unison_inplace(a, b):
        assert len(a) == len(b)
        p = numpy.random.permutation(len(a))
        return a[p], b[p]

    [training_tensors[0],training_tensors[1]] = shuffle_in_unison_inplace(training_tensors[0],training_tensors[1])
    [validation_tensors[0], validation_tensors[1]] = shuffle_in_unison_inplace(validation_tensors[0], validation_tensors[1])
    '''
    [training_tensors[0],training_tensors[1]] = shuffle(training_tensors[0],training_tensors[1])
    [validation_tensors[0], validation_tensors[1]] = shuffle(validation_tensors[0], validation_tensors[1])

training_set, validation_set = [Dataset(tensors=t,
                                        names=training_set.names,
                                        formats=training_set.formats)
                                for t in (training_tensors, validation_tensors)]

training_iter = training_set.iterator(iterator_type='sequential', batch_size=100)

image_node, label_node = training_iter.make_input_nodes()

float_image_node = RescaleImage(image_node)

input_shape = float_image_node.output_format.shape
conv_input_node = FormatNode(input_node=float_image_node,  # axis order: batch, rows, cols
                             output_format=DenseFormat(axes=('b', 'c', '0', '1'),  # batch, channels, rows, cols
                                                       shape=(input_shape[0],  # batch size (-1)
                                                              1,               # num. channels
                                                              input_shape[1],  # num. rows (28)
                                                              input_shape[2]), # num cols (28)
                                                       dtype=None),  # don't change the input's dtype
                             axis_map={'b': ('b', 'c')})  # split batch axis into batch & channel axes

layers = [conv_input_node]

for _ in range(2):  # repeat twice
    layers.append(AffineLayer(input_node=layers[-1],  # last element of <layers>
                              output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                        shape=(-1, 10),   # output shape: (variable batch size, 10 classes)
                                                        dtype=None)    # don't change the input data type
                              ))

layers.append(SoftmaxLayer(input_node=layers[-1],
                           output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                     shape=(-1, 10),   # output shape: (variable batch size, 10 classes)
                                                     dtype=None),      # don't change the input data type
                           ))  # collapse the channel, row, and column axes to a single feature axis

rng = numpy.random.RandomState(389323)  # mash the keypad with your forehead to come up with a suitable seed
softmax_layer = layers[-1]
affine_weights_symbol = softmax_layer.affine_node.linear_node.params
affine_weights_values = affine_weights_symbol.get_value()
std_deviation = .05
affine_weights_values[...] = rng.standard_normal(affine_weights_values.shape) * std_deviation
affine_weights_symbol.set_value(affine_weights_values)

for i in range(1,3):
    rng = numpy.random.RandomState(346523)  # mash the keypad with your forehead to come up with a suitable seed
    affine_layer = layers[i]
    affine_weights_symbol = affine_layer.affine_node.linear_node.params
    affine_weights_values = affine_weights_symbol.get_value()
    std_deviation = .05
    affine_weights_values[...] = rng.standard_normal(affine_weights_values.shape) * std_deviation
    affine_weights_symbol.set_value(affine_weights_values)

loss_node = CrossEntropy(softmax_layer, label_node)

# Remember parameters at current (y_(j,t))
param_symbols = []

# add the filters and biases from each convolutional layer
for i in range(1,4):
    param_symbols.append(layers[i].affine_node.linear_node.params)
    param_symbols.append(layers[i].affine_node.bias_node.params)

# Set up the symbolic old parameter to calculate the gradient with respect to it (x_j)
old_param_symbols = []
for i in range(len(param_symbols)):
    old_param_symbols.append(theano.shared(numpy.zeros(param_symbols[i].get_value().shape, dtype=theano.config.floatX)))

# Create symbolic expression for gradient at current parameters
scalar_loss_symbol = loss_node.output_symbol.mean()  # the mean over the batch axis. Very important not to use sum().
gradient_symbols = [theano.gradient.grad(scalar_loss_symbol, p) for p in param_symbols]  # derivatives of loss w.r.t. each of the params

# Create symbolic expression for gradient at old parameter (alter existing gradient symbols with theano.clone)
scalar_loss_symbol2 = theano.clone(scalar_loss_symbol, replace = {param_symbols[i]: old_param_symbols[i] for i in range(len(param_symbols))} )

gradient_loss_symbols_at_old_params = [theano.gradient.grad(scalar_loss_symbol2, p) for p in old_param_symbols]


# Create parameter updaters:
param_updaters_S2GD_rolling = [SemiSgdParameterUpdater(parameter=param_symbol,
                                      gradient=gradient_symbol,
                                      gradient_at_old_params = gradient_loss_symbol_at_old_params,
                                      learning_rate=0.02,
                                      momentum=0.8,
                                      method = 'SGD',
                                      use_nesterov=False)
                  for param_symbol, gradient_symbol, gradient_loss_symbol_at_old_params
                  in safe_izip(param_symbols, gradient_symbols, gradient_loss_symbols_at_old_params)]
# packages chain of nodes from the uint8 image_node up to the softmax_layer, to be saved to a file.
model = SerializableModel([image_node], [softmax_layer])

# A Node that outputs 1 if output_node's label diagrees with label_node's label, 0 otherwise.
misclassification_node = Misclassification(softmax_layer, label_node)

#
# Callbacks to feed the misclassification rate (MCR) to after each epoch:
#

# Prints misclassificiation rate (must be a module-level function to be pickleable).
def print_misclassification_rate(values, _):  # ignores 2nd argument (formats)
    print("Misclassification rate: %s" % str(values))

# Saves <model> to file "best_model.pkl" if MCR is the best yet seen.
saves_best = SavesAtMinimum(model, "./best_model.pkl")

# Raises a StopTraining exception if MCR doesn't decrease for more than 10 epochs.
training_stopper = StopsOnStagnation(max_epochs=400, min_proportional_decrease=0.0)

# Measures the average misclassification rate over some dataset
misclassification_rate_monitor = AverageMonitor(misclassification_node.output_symbol,
                                                misclassification_node.output_format,
                                                callbacks=[print_misclassification_rate,
                                                           saves_best,
                                                           training_stopper])

validation_iter = validation_set.iterator(iterator_type='sequential', batch_size=10)

# Gets called by trainer between training epochs.
validation_callback = ValidationCallback(inputs=[image_node.output_symbol, label_node.output_symbol],
                                         input_iterator=validation_iter,
                                         monitors=[misclassification_rate_monitor])


trainer_S2GD_rolling = SemiSgd(inputs=[image_node, label_node],
                  input_iterator=training_iter,
                  parameters=param_symbols,
                  old_parameters=old_param_symbols,
                  parameter_updaters=param_updaters_S2GD_rolling,
                  gradient=gradient_symbols,
                  monitors=[],
                  training_set = training_set,
                  epoch_callbacks=[validation_callback,  # measure validation misclassification rate, quit if it stops falling
                                   LimitsNumEpochs(10)])  # perform no more than 100 epochs


print "Training S2GD_roll: "
start_time = time.time()
_classification_errors_S2GD_rolling = trainer_S2GD_rolling.train()
print _classification_errors_S2GD_rolling
elapsed_time_S2GD_rolling = time.time() - start_time

# Plot the results in one figure:
plt.plot(_classification_errors_S2GD_rolling)
plt.legend(['S2GD_roll'])
plt.title('Learning Curves (Learning rate = 0.02, Momentum = 0.8)')
plt.xlabel('Epochs')
plt.ylabel('Classification error')
plt.show()

print "The time elapsed for training is: "
print "For S2GD_roll: ", elapsed_time_S2GD_rolling