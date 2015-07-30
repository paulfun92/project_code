import os
import argparse
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
from nose.tools import (assert_true,
                        assert_is_instance,
                        assert_greater,
                        assert_greater_equal,
                        assert_less_equal,
                        assert_equal)
from simplelearn.nodes import (Node,
                               AffineLayer,
                               CastNode,
                               ReLU,
                               Dropout,
                               CrossEntropy,
                               Misclassification,
                               SoftmaxLayer,
                               RescaleImage,
                               FormatNode)
from simplelearn.utils import safe_izip
from simplelearn.asserts import (assert_all_greater,
                                 assert_all_less_equal,
                                 assert_all_integer)
from simplelearn.io import SerializableModel
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  Sgd,
                                  LogsToLists,
                                  SavesAtMinimum,
                                  Monitor,
                                  AverageMonitor,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation)
import pdb
from extension_BGFS import Bgfs, BgfsParameterUpdater, Bgfs2
import time


###### HERE IS THE MAIN EXAMPLE ##########

training_set, testing_set = load_mnist()

training_tensors = [t[:50000, ...] for t in training_set.tensors]  # the first 50000 examples
validation_tensors = [t[50000:, ...] for t in training_set.tensors]  # the remaining 10000 examples

shuffle_dataset = True
if shuffle_dataset == True:
    def shuffle_in_unison_inplace(a, b):
        assert len(a) == len(b)
        p = numpy.random.permutation(len(a))
        return a[p], b[p]

    [training_tensors[0],training_tensors[1]] = shuffle_in_unison_inplace(training_tensors[0],training_tensors[1])
    [validation_tensors[0], validation_tensors[1]] = shuffle_in_unison_inplace(validation_tensors[0], validation_tensors[1])

training_set, validation_set = [Dataset(tensors=t,
                                        names=training_set.names,
                                        formats=training_set.formats)
                                for t in (training_tensors, validation_tensors)]

input_size = training_tensors[0].shape[1]*training_tensors[0].shape[2]
sizes = [10,10,10]

training_iter = training_set.iterator(iterator_type='sequential', batch_size=50000)

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

rng = numpy.random.RandomState(281934)
std_deviation = .05

params_temp1 = [rng.standard_normal( (input_size * sizes[0]) ).astype(theano.config.floatX)*std_deviation,
                numpy.zeros(sizes[0], dtype=theano.config.floatX) ]

params_temp2 = sum([ [rng.standard_normal( sizes[i] * sizes[i+1] ).astype(theano.config.floatX)*std_deviation,
                      numpy.zeros(sizes[i+1], dtype=theano.config.floatX)] for i in range(len(sizes)-1) ],[] )

params_flat_values = numpy.concatenate( params_temp1 + params_temp2 )

params_flat = theano.shared(params_flat_values)
params_old_flat = theano.shared(params_flat_values)

param_arrays = []
index_to = input_size * sizes[0]
param_arrays.append(params_flat[:index_to].reshape((input_size, sizes[0]))) # Add weights
index_from = index_to
index_to += sizes[0]
param_arrays.append(params_flat[index_from:index_to]) # Add bias

for i in range(len(sizes)-1):

    index_from = index_to
    index_to += sizes[i]*sizes[i+1]
    param_arrays.append(params_flat[index_from:index_to].reshape((sizes[i], sizes[i+1]))) # Add weight
    #print(index_from, index_to)
    #print 'reshaped to'
    #print(sizes[i], sizes[i+1])
    index_from = index_to
    index_to += sizes[i+1]
    param_arrays.append(params_flat[index_from:index_to]) # Add bias

layers = [conv_input_node]

for i in range(len(sizes)-1):  # repeat twice
    layers.append(AffineLayer(input_node=layers[-1],  # last element of <layers>
                              output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                        shape=(-1, sizes[i]),   # output shape: (variable batch size, 10 classes)
                                                        dtype=None) ,   # don't change the input data type
                              weights = param_arrays[i*2],
                              bias = param_arrays[i*2+1]
                              ))

layers.append(SoftmaxLayer(input_node=layers[-1],
                           output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                     shape=(-1, sizes[i+1]),   # output shape: (variable batch size, 10 classes)
                                                     dtype=None),      # don't change the input data type
                           weights = param_arrays[(i+1)*2],
                           bias = param_arrays[(i+1)*2+1]
                           ))  # collapse the channel, row, and column axes to a single feature axis

softmax_layer = layers[-1]
#correct_output = theano.tensor.scalar(dtype=theano.config.floatX)
loss_node = CrossEntropy(softmax_layer, label_node)

scalar_loss_symbol = loss_node.output_symbol.mean()  # the mean over the batch axis. Very important not to use sum().
scalar_loss_symbol2 = theano.clone(scalar_loss_symbol, replace = {params_flat: params_old_flat})

gradient_symbol = theano.gradient.grad(scalar_loss_symbol, params_flat)
gradient_symbol_old_params = theano.gradient.grad(scalar_loss_symbol2, params_old_flat)
#hessian_symbol = theano.gradient.hessian(scalar_loss_symbol, params_flat)

# For simplicity, we won't use Nesterov accelerated gradients for this example.
param_updater = BgfsParameterUpdater(parameters=params_flat,
                                     old_parameters=params_old_flat,
                                     gradient=gradient_symbol,
                                     gradient_at_old_params=gradient_symbol_old_params,
                                     learning_rate=.3)

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
training_stopper = StopsOnStagnation(max_epochs=10, min_proportional_decrease=0.0)

# Measures the average misclassification rate over some dataset
misclassification_rate_monitor = AverageMonitor(misclassification_node.output_symbol,
                                                misclassification_node.output_format,
                                                callbacks=[print_misclassification_rate,
                                                           saves_best,
                                                           training_stopper])

validation_iter = validation_set.iterator(iterator_type='sequential', batch_size=100)

# Gets called by trainer between training epochs.
validation_callback = ValidationCallback(inputs=[image_node.output_symbol, label_node.output_symbol],
                                         input_iterator=validation_iter,
                                         monitors=[misclassification_rate_monitor])

trainer = Bgfs2([image_node, label_node],
              training_iter,
              params_flat,
              param_updater,
              monitors=[],
              training_set = training_set,
              epoch_callbacks=[validation_callback,  # measure validation misclassification rate, quit if it stops falling
                               LimitsNumEpochs(100)])  # perform no more than 100 epochs

start_time = time.time()
_classification_errors = trainer.train()
print _classification_errors
elapsed_time = time.time() - start_time

plt.plot(_classification_errors)
plt.show()

print "The time elapsed for training is ", elapsed_time

