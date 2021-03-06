#! /usr/bin/env python

import os
import sys
sys.dont_write_bytecode = True
import argparse
import numpy
import timeit
import time
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from nose.tools import (assert_true,
                        assert_is_instance,
                        assert_greater,
                        assert_greater_equal,
                        assert_less_equal,
                        assert_equal)
from simplelearn.nodes import (Node,
                               AffineLayer,
                               CastNode,
                               Dropout,
                               CrossEntropy,
                               Misclassification,
                               SoftmaxLayer,
                               RescaleImage)
from simplelearn.utils import safe_izip
from simplelearn.asserts import (assert_all_greater,
                                 assert_all_less_equal,
                                 assert_all_integer)
from simplelearn.io import SerializableModel
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (# LogsToLists,
                                  SavesAtMinimum,
                                  MeanOverEpoch,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  # PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation,
                                  EpochLogger,
                                  EpochCallback,
                                  EpochTimer2)
import pdb

cwd = os.getcwd()
sys.path.append(cwd + '/extensions')
from extension_GD import Bgfs

class ImageLookeupNode(Node):

    def __init__(self,input_node, images_array):

        self.images = images_array
        output_symbol = self.images[input_node.output_symbol]
        output_format = DenseFormat(axes=('b', '0', '1'),
                                    shape=(-1, 28, 28),
                                    dtype='uint8')

        super(ImageLookeupNode, self).__init__(input_nodes=input_node,
                                        output_symbol=output_symbol,
                                        output_format=output_format)

class LabelLookeupNode(Node):

    def __init__(self,input_node, labels_array):

        self.labels = labels_array
        output_symbol = self.labels[input_node.output_symbol]
        output_format = DenseFormat(axes=('b', ),
                                    shape=(-1, ),
                                    dtype='uint8')

        super(LabelLookeupNode, self).__init__(input_nodes=input_node,
                                        output_symbol=output_symbol,
                                        output_format=output_format)

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Trains multilayer perceptron to classify MNIST digits. "
                     "Default arguments are the ones used by Pylearn2's mlp "
                     "tutorial #3."))

    def positive_float(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        return result

    def non_negative_float(arg):
        result = float(arg)
        assert_greater_equal(result, 0.0)
        return result

    def non_negative_int(arg):
        result = int(arg)
        assert_greater_equal(result, 0)
        return result

    def positive_int(arg):
        result = int(arg)
        assert_greater(result, 0)
        return result

    def legit_prefix(arg):
        abs_path = os.path.abspath(arg)
        assert_true(os.path.isdir(os.path.split(abs_path)[0]))
        assert_equal(os.path.splitext(abs_path)[1], "")
        return arg

    def positive_0_to_1(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        assert_less_equal(result, 1.0)
        return result

    parser.add_argument("--output-prefix",
                        type=legit_prefix,
                        required=False,
                        default='output',
                        help=("Directory and optional prefix of filename to "
                              "save the log to."))

    # All but one of the default hyperparameter values below are taken from
    # Pylearn2, in pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml
    #
    # The one exception is final-momentum, which was changed from .99 to
    # .5. (.99 led to divergence, which is weird).
    #
    # non-default arguments that work for me:
    #
    # --dropout-include-rates 1 1 .5 --learning-rate .001
    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=1,
                        help=("Learning rate."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=50000,  # full batch method normally
                        help="batch size")

    parser.add_argument("--batch-size-calculation",
                        type=non_negative_int,
                        default=50000,
                        help="batch size used for calculating the gradient")

    parser.add_argument("--no-shuffle-dataset",
                        default=False,
                        action="store_true",
                        help=("Shuffle the dataset before use"))

    parser.add_argument("--dropout-include-rates",
                        default=(1.0, 1.0, 1.0),  # i.e. no dropout
                        type=positive_0_to_1,
                        nargs=3,
                        help=("The dropout include rates for the outputs of "
                              "the first two layers. Must be in the range "
                              "(0.0, 1.0]. If 1.0, the Dropout node will "
                              "simply be omitted. For no dropout, use "
                              "1.0 1.0 (this is the default). Make sure to "
                              "lower the learning rate when using dropout. "
                              "I'd suggest a learning rate of 0.001 for "
                              "dropout-include-rates of 0.5 0.5."))

    parser.add_argument("--validation-size",
                        type=non_negative_int,
                        default=10000,
                        metavar="V",
                        help=("Hold out the last V examples of the training "
                              "set to use as a validation set. If V is 0, "
                              "use the MNIST test set as the validation set."))

    parser.add_argument("--armijo",
                        default=True,
                        action="store_true",
                        help="use Armijo line search procedure.")

    parser.add_argument("--tangent",
                        default=False,
                        action="store_true",
                        help="Use weight norm regularization by subtracting off the non-tangent component of the weight updates")

    return parser.parse_args()

class EpochTimer(EpochCallback):
    '''
    Prints the epoch number and duration after each epoch.
    '''

    def __init__(self):
        self.start_time = None
        self.epoch_number = None

    def on_start_training(self):
        self.start_time = timeit.default_timer()
        self.epoch_number = 0

    def on_epoch(self):
        end_time = timeit.default_timer()

        print("Epoch {} duration: {}".format(self.epoch_number,
                                             end_time - self.start_time))

        self.start_time = end_time
        self.epoch_number += 1


def build_fc_classifier(input_node,
                        sizes,
                        sparse_init_counts,
                        dropout_include_probabilities,
                        rng,
                        theano_rng):
    '''
    Builds a stack of fully-connected layers followed by a Softmax.

    Each hidden layer will be preceded by a ReLU.

    Initialization:

    Weights are initialized in the same way as in Pylearn2's MLP tutorial:
    pylearn2/scripts/tutorials/multilayer_perceptron/mlp_tutorial_part_3.yaml

    This means the following:

    Of the N affine layers, the weights of the first N-1 are to all 0.0, except
    for k randomly-chosen elements, which are set to some random number drawn
    from the normal distribution with stddev=1.0.

    The biases are all initialized to 0.0.
    The last layer's weights and biases are both set to 0.0.

    Parameters
    ----------
    input_node: Node
      The node to build the stack on.

    sizes: Sequence
      A sequence of ints, indicating the output sizes of each layer.
      The last int is the number of classes.

    sparse_init_counts:
      A sequence of N-1 ints, where N = len(sizes).
      Used to initialize the weights of the first N-1 layers.
      If the n'th element is x, this means that the n'th layer
      will have x nonzeros, with the rest initialized to zeros.

    dropout_include_probabilities: Sequence
      A Sequence of N-1 floats, where N := len(sizes)
      The dropout include probabilities for the outputs of each of the layers,
      except for the final one.
      If any of these probabilities is 1.0, the corresponding Dropout node
      will be omitted.

    rng: numpy.random.RandomState
      The RandomState to draw initial weights from.

    theano_rng: theano.tensor.shared_randomstreams.RandomStreams
      The RandomStreams to draw dropout masks from.

    Returns
    -------
    rval: tuple
      (affine_nodes, output_node), where affine_nodes is a list of the
      AffineNodes, in order, and output_node is the final node, a Softmax.
    '''
    assert_is_instance(input_node, Node)

    # pylint: disable=no-member
    assert_equal(input_node.output_format.dtype,
                 numpy.dtype(theano.config.floatX))

    assert_greater(len(sizes), 0)
    assert_all_greater(sizes, 0)

    assert_equal(len(sparse_init_counts), len(sizes) - 1)
    assert_all_integer(sparse_init_counts)
    assert_all_greater(sparse_init_counts, 0)
    assert_all_less_equal(sparse_init_counts, sizes[:-1])

    assert_equal(len(dropout_include_probabilities), len(sizes))

    '''
    affine_nodes = []

    last_node = input_node

    for layer_index, layer_output_size in enumerate(sizes):
        # Add dropout, if asked for
        include_probability = dropout_include_probabilities[layer_index]
        if include_probability != 1.0:
            last_node = Dropout(last_node, include_probability, theano_rng)

        output_format = DenseFormat(axes=('b', 'f'),
                                    shape=(-1, layer_output_size),
                                    dtype=None)

        if layer_index < (len(sizes) - 1):
            last_node = AffineLayer(last_node, output_format)
        else:
            last_node = SoftmaxLayer(last_node, output_format)

        affine_nodes.append(last_node.affine_node)


    # Not used in this demo, but keeping it in in case we want to start using
    # it again.
    def init_sparse_bias(shared_variable, num_nonzeros, rng):

        #Mimics the sparse initialization in
        #pylearn2.models.mlp.Linear.set_input_space()


        params = shared_variable.get_value()
        assert_equal(params.shape[0], 1)

        assert_greater_equal(num_nonzeros, 0)
        assert_less_equal(num_nonzeros, params.shape[1])

        params[...] = 0.0

        indices = rng.choice(params.size,
                             size=num_nonzeros,
                             replace=False)

        # normal dist with stddev=1.0
        params[0, indices] = rng.randn(num_nonzeros)

        # Found that for biases, this didn't help (it increased the
        # final misclassification rate by .001)
        # if num_nonzeros > 0:
        #     params /= float(num_nonzeros)

        shared_variable.set_value(params)

    def init_sparse_linear(shared_variable, num_nonzeros, rng):
        params = shared_variable.get_value()
        params[...] = 0.0

        assert_greater_equal(num_nonzeros, 0)
        assert_less_equal(num_nonzeros, params.shape[0])

        for c in xrange(params.shape[1]):
            indices = rng.choice(params.shape[0],
                                 size=num_nonzeros,
                                 replace=False)

            # normal dist with stddev=1.0, divided by 255.0
            #
            # We need to divide by 255 for convergence. This is because
            # we're using unnormalized (i.e. 0 to 255) pixel values, unlike the
            # 0.0-to-1.0 pixels in
            # pylearn2.scripts.tutorials.multilayer_perceptron/
            #
            # We could just do as the above tutorial does and normalize the
            # pixels to [0.0, 1.0], and not rescale the weights. However,
            # experiments show that this converges to a higher error, and also
            # makes mnist_visualizer.py's results look very "staticky", without
            # any recognizable digit hallucinations.
            params[indices, c] = rng.randn(num_nonzeros) / 255.0

        shared_variable.set_value(params)

    # Initialize the affine layer weights (not the biases, and not the softmax
    # weights)
    for sparse_init_count, affine_node in safe_izip(sparse_init_counts,
                                                    affine_nodes[:-1]):
        # pylearn2 doesn't sparse_init the biases. I also found that
        # doing so slightly increases the final misclassification rate.
        init_sparse_linear(affine_node.linear_node.params,
                           sparse_init_count,
                           rng)

    #################################################################################################
    ### BUILD THE SECOND NETWORK WITH FLAT PARAMETERS (given the dimensions of the first) ###########
    #################################################################################################

    parameters = []
    shapes = []
    for affine_node in affine_nodes:
        weights = affine_node.linear_node.params
        bias = affine_node.bias_node.params
        parameters.append(weights)
        parameters.append(bias)
        shapes.append(weights.get_value().shape)
        shapes.append(bias.get_value().shape)

    params_flat_values = numpy.asarray([], dtype=theano.config.floatX)
    for parameter in parameters:
        vector_param = numpy.asarray(numpy.ndarray.flatten(parameter.get_value()), dtype=theano.config.floatX)
        params_flat_values = numpy.append(params_flat_values, vector_param)

    params_flat = theano.shared(params_flat_values)
    params_old_flat = theano.shared(params_flat_values)

    affine_nodes = []
    last_node = input_node
    counter = 0
    index_from = 0
    for layer_index, layer_output_size in enumerate(sizes):

        shape1 = shapes[counter]
        shape2 = shapes[counter+1]
        size1= numpy.prod(numpy.asarray(shape1))
        size2= numpy.prod(numpy.asarray(shape2))
        index_to = index_from + size1
        weights_ = params_flat[index_from:index_to].reshape(shape1)
        index_from = index_to
        index_to = index_from + size2
        bias_ = params_flat[index_from:index_to].reshape(shape2)
        index_from = index_to
        counter = counter + 2

        # Add dropout, if asked for
        include_probability = dropout_include_probabilities[layer_index]
        if include_probability != 1.0:
            last_node = Dropout(last_node, include_probability, theano_rng)

        output_format = DenseFormat(axes=('b', 'f'),
                                    shape=(-1, layer_output_size),
                                    dtype=None)

        if layer_index < (len(sizes) - 1):
            last_node = AffineLayer(last_node, output_format, weights=weights_, bias=bias_)
        else:
            last_node = SoftmaxLayer(last_node, output_format, weights=weights_, bias=bias_)

        affine_nodes.append(last_node.affine_node)

    return affine_nodes, last_node, params_flat, params_old_flat
    '''

    std_deviation = .05

    input_size = 784
    params_temp1 = [rng.standard_normal( (sizes[0]* input_size) ).astype(theano.config.floatX)*std_deviation,
                    numpy.zeros(sizes[0], dtype=theano.config.floatX) ]

    params_temp2 = sum([ [rng.standard_normal( sizes[i] * sizes[i+1] ).astype(theano.config.floatX)*std_deviation,
                          numpy.zeros(sizes[i+1], dtype=theano.config.floatX)] for i in range(len(sizes)-1) ],[] )

    params_flat_values = numpy.concatenate( params_temp1 + params_temp2 )

    params_flat = theano.shared(params_flat_values)
    params_old_flat = theano.shared(params_flat_values)

    shapes = []
    param_arrays = []
    index_to = input_size * sizes[0]
    param_arrays.append(params_flat[:index_to].reshape((sizes[0], input_size))) # Add weights
    shapes.append((input_size, sizes[0]))
    index_from = index_to
    index_to += sizes[0]
    param_arrays.append(params_flat[index_from:index_to]) # Add bias
    shapes.append((index_to-index_from, ))

    for i in range(len(sizes)-1):

        index_from = index_to
        index_to += sizes[i]*sizes[i+1]
        param_arrays.append(params_flat[index_from:index_to].reshape((sizes[i+1],sizes[i]))) # Add weight
        shapes.append((sizes[i], sizes[i+1]))
        #print(index_from, index_to)
        #print 'reshaped to'
        #print(sizes[i], sizes[i+1])
        index_from = index_to
        index_to += sizes[i+1]
        param_arrays.append(params_flat[index_from:index_to]) # Add bias
        shapes.append((index_to-index_from, ))

    layers = [input_node]

    for i in range(len(sizes)-1):  # repeat twice
        layers.append(AffineLayer(input_node=layers[-1],  # last element of <layers>
                                  output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                            shape=(-1, sizes[i]),   # output shape: (variable batch size, 10 classes)
                                                            dtype=None) ,   # don't change the input data type
                                  weights = theano.tensor.transpose(param_arrays[i*2]),
                                  bias = param_arrays[i*2+1]
                                  ))

    layers.append(SoftmaxLayer(input_node=layers[-1],
                               output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                         shape=(-1, sizes[i+1]),   # output shape: (variable batch size, 10 classes)
                                                         dtype=None),      # don't change the input data type
                               weights = theano.tensor.transpose(param_arrays[(i+1)*2]),
                               bias = param_arrays[(i+1)*2+1]
                               ))  # collapse the channel, row, and column axes to a single feature axis

    softmax_layer = layers[-1]

    last_node = softmax_layer
    affine_nodes = []
    for i in range(1,len(layers)):
        affine_nodes.append(layers[i].affine_node)

    print shapes

    return affine_nodes, last_node, params_flat, params_old_flat, shapes


def print_loss(values, _):  # 2nd argument: formats
    print("Average loss: %s" % str(values))


def print_feature_vector(values, _):
    print("Average feature vector: %s" % str(values))


def print_mcr(values, _):
    print("Misclassification rate: %s" % str(values))


def main():
    args = parse_args()

    # Hyperparameter values taken from Pylearn2:
    # In pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml

    sizes = [500, 500, 10]
    sparse_init_counts = [15, 15]
    assert_equal(len(sparse_init_counts), len(sizes) - 1)

    assert_equal(sizes[-1], 10)

    mnist_training, mnist_testing = load_mnist()

    # split training set into training and validation sets
    tensors = mnist_training.tensors
    training_tensors = [t[:-args.validation_size, ...] for t in tensors]
    validation_tensors = [t[-args.validation_size:, ...] for t in tensors]

    if args.no_shuffle_dataset == False:
        def shuffle_in_unison_inplace(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        [training_tensors[0],training_tensors[1]] = shuffle_in_unison_inplace(training_tensors[0],training_tensors[1])
        [validation_tensors[0], validation_tensors[1]] = shuffle_in_unison_inplace(validation_tensors[0], validation_tensors[1])

    all_images_shared = theano.shared(numpy.vstack([training_tensors[0],validation_tensors[0]]))
    all_labels_shared = theano.shared(numpy.concatenate([training_tensors[1],validation_tensors[1]]))

    length_training = training_tensors[0].shape[0]
    length_validation = validation_tensors[0].shape[0]
    indices_training = numpy.asarray(range(length_training))
    indices_validation = numpy.asarray(range(length_training, length_training + length_validation))
    indices_training_dataset = Dataset( tensors=[indices_training], names=['indices'], formats=[DenseFormat(axes=['b'],shape=[-1],dtype='int64')] )
    indices_validation_dataset = Dataset( tensors=[indices_validation], names=['indices'], formats=[DenseFormat(axes=['b'],shape=[-1],dtype='int64')] )
    indices_training_iterator = indices_training_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size_calculation)
    indices_validation_iterator = indices_validation_dataset.iterator(iterator_type='sequential',batch_size=10000)

    mnist_validation_iterator = indices_validation_iterator
    mnist_training_iterator = indices_training_iterator

    input_indices_symbolic, = indices_training_iterator.make_input_nodes()
    image_lookup_node = ImageLookeupNode(input_indices_symbolic, all_images_shared)
    label_lookup_node = LabelLookeupNode(input_indices_symbolic, all_labels_shared)

    #image_node = CastNode(image_lookup_node, 'floatX')
    image_node = RescaleImage(image_lookup_node)
    # image_node = RescaleImage(image_uint8_node)

    rng = numpy.random.RandomState(281934)
    theano_rng = RandomStreams(23845)

    (affine_nodes,
     output_node,
     params_flat,
     params_old_flat,
     shapes) = build_fc_classifier(image_node,
                                        sizes,
                                        sparse_init_counts,
                                        args.dropout_include_rates,
                                        rng,
                                        theano_rng)

    loss_node = CrossEntropy(output_node, label_lookup_node)
    loss_sum = loss_node.output_symbol.mean()
    max_epochs = 300
    gradient = theano.gradient.grad(loss_sum, params_flat)

    #
    # Makes batch and epoch callbacks
    #

    '''
    def make_output_basename(args):
        assert_equal(os.path.splitext(args.output_prefix)[1], "")
        if os.path.isdir(args.output_prefix) and \
           not args.output_prefix.endswith('/'):
            args.output_prefix += '/'

        output_dir, output_prefix = os.path.split(args.output_prefix)
        if output_prefix != "":
            output_prefix = output_prefix + "_"

        output_prefix = os.path.join(output_dir, output_prefix)

        return "{}lr-{}_mom-{}_nesterov-{}_bs-{}".format(
            output_prefix,
            args.learning_rate,
            args.initial_momentum,
            args.nesterov,
            args.batch_size)
    '''

    assert_equal(os.path.splitext(args.output_prefix)[1], "")
    if os.path.isdir(args.output_prefix) and \
       not args.output_prefix.endswith('/'):
        args.output_prefix += '/'

    output_dir, output_prefix = os.path.split(args.output_prefix)
    if output_prefix != "":
        output_prefix = output_prefix + "_"

    output_prefix = os.path.join(output_dir, output_prefix)

    epoch_logger = EpochLogger(output_prefix + "GD.h5")


    # misclassification_node = Misclassification(output_node, label_node)
    # mcr_logger = LogsToLists()
    # training_stopper = StopsOnStagnation(max_epochs=10,
    #                                      min_proportional_decrease=0.0)

    misclassification_node = Misclassification(output_node, label_lookup_node)

    validation_loss_monitor = MeanOverEpoch(loss_node, callbacks=[])
    epoch_logger.subscribe_to('validation mean loss', validation_loss_monitor)

    validation_misclassification_monitor = MeanOverEpoch(
        misclassification_node,
        callbacks=[print_mcr,
                   StopsOnStagnation(max_epochs=20,
                                     min_proportional_decrease=0.0)])

    epoch_logger.subscribe_to('validation misclassification',
                              validation_misclassification_monitor)

    # batch callback (monitor)
    # training_loss_logger = LogsToLists()
    training_loss_monitor = MeanOverEpoch(loss_node, callbacks=[print_loss])
    epoch_logger.subscribe_to('training mean loss', training_loss_monitor)

    training_misclassification_monitor = MeanOverEpoch(misclassification_node,
                                                       callbacks=[])
    epoch_logger.subscribe_to('training misclassification %',
                              training_misclassification_monitor)

    # epoch callbacks
    # validation_loss_logger = LogsToLists()


    def make_output_filename(args, best=False):
        basename = make_output_basename(args)
        return "{}{}.pkl".format(basename, '_best' if best else "")

    #model = SerializableModel([input_indices_symbolic], [output_node])
    #saves_best = SavesAtMinimum(model, make_output_filename(args, best=True))

    validation_loss_monitor = MeanOverEpoch(
        loss_node,
        callbacks=[])

    epoch_logger.subscribe_to('validation loss', validation_loss_monitor)

    epoch_timer = EpochTimer2()
    epoch_logger.subscribe_to('epoch duration', epoch_timer)

    validation_callback = ValidationCallback(
        inputs=[input_indices_symbolic.output_symbol],
        input_iterator=mnist_validation_iterator,
        epoch_callbacks=[validation_loss_monitor,
                         validation_misclassification_monitor])

    trainer = Bgfs(inputs=[input_indices_symbolic],
                  parameters=params_flat,
                  gradient=gradient,
                  learning_rate=args.learning_rate,
                  training_iterator=mnist_training_iterator,
                  validation_iterator=mnist_validation_iterator,
                  scalar_loss=loss_sum,
                  armijo=args.armijo,
                  tangent=args.tangent,
                  batch_size=args.batch_size,
                  epoch_callbacks=([
                             #training_loss_monitor,
                             # training_misclassification_monitor,
                              validation_callback,
                              LimitsNumEpochs(max_epochs),
                              epoch_timer]),
                  param_shapes=shapes)

                                                   # validation_loss_monitor]))

    # stuff_to_pickle = OrderedDict(
    #     (('model', model),
    #      ('validation_loss_logger', validation_loss_logger)))

    # Pickling the trainer doesn't work when there are Dropout nodes.
    # stuff_to_pickle = OrderedDict(
    #     (('trainer', trainer),
    #      ('validation_loss_logger', validation_loss_logger),
    #      ('model', model)))

    # trainer.epoch_callbacks += (momentum_updaters +
    #                             [PicklesOnEpoch(stuff_to_pickle,
    #                                             make_output_filename(args),
    #                                             overwrite=False),
    #                              validation_callback,
    #                              LimitsNumEpochs(max_epochs)])

    loss_function = theano.function([input_indices_symbolic.output_symbol],loss_sum)
    cost_args = mnist_training_iterator.next()
    print loss_function(*cost_args)

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print("Total elapsed time is for training is: ", elapsed_time)


if __name__ == '__main__':
    main()
