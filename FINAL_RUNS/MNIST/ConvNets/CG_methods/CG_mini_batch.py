#! /usr/bin/env python

'''
Demonstrates training a convolutional net on MNIST.
'''

from __future__ import print_function

import os
import sys
sys.dont_write_bytecode = True
import time
import argparse
import timeit
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
from simplelearn.nodes import (Conv2dLayer,
                               Dropout,
                               CrossEntropy,
                               Misclassification,
                               SoftmaxLayer,
                               RescaleImage,
                               FormatNode,
                               Node)
from simplelearn.utils import safe_izip
from simplelearn.asserts import (assert_floating,
                                 assert_all_equal,
                                 assert_all_greater,
                                 assert_all_integer)
from simplelearn.io import SerializableModel
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (limit_param_norms,
                                  LogsToLists,
                                  SavesAtMinimum,
                                  MeanOverEpoch,
                                  EpochCallback,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation,
                                  EpochLogger,
                                  EpochTimer2)

cwd = os.getcwd()
sys.path.append(cwd + '/extensions')
from extension_CG_restrict_norm3_armijo_inner_loop import CG


def parse_args():
    '''
    Parses the command-line args.
    '''

    parser = argparse.ArgumentParser(
        description=("Trains multilayer perceptron to classify MNIST digits. "
                     "Default arguments are the ones used by Pylearn2's mlp "
                     "tutorial #3."))

    # pylint: disable=missing-docstring

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

    def positive_0_to_1(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        assert_less_equal(result, 1.0)
        return result

    def legit_prefix(arg):
        abs_path = os.path.abspath(arg)
        assert_true(os.path.isdir(os.path.split(abs_path)[0]))
        assert_equal(os.path.splitext(abs_path)[1], "")
        return arg

    def non_negative_0_to_1(arg):
        result = float(arg)
        assert_greater_equal(result, 0.0)
        assert_less_equal(result, 1.0)
        return result

    def max_norm_arg(arg):
        arg = float(arg)
        if arg < 0.0:
            return numpy.inf
        else:
            assert_greater(arg, 0.0)
            return arg

    # This checks if the method is entered correctly
    def correct_method(arg):
        ''' Arg checker for input of method choice'''
        arg = str(arg)
        if arg != 'FR' and arg != 'PR' and arg != 'PR_plus' and arg != 'HS' and arg != 'DY':
            raise ValueError("Please enter a valid choice for optimization method: 'FR', 'PR', 'HS' or 'DY'"
                             % arg)
        return arg

    parser.add_argument("--output-prefix",
                        type=legit_prefix,
                        required=False,
                        default='output',
                        help=("Directory and optional prefix of filename to "
                              "save the log to."))

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=1,  # .01 used in pylearn2 demo
                        help=("Learning rate."))

    parser.add_argument("--no-shuffle-dataset",
                        default=False,  # original didn't use nesterov
                        action="store_true",
                        help=("Shuffle dataset before running "
                              "(default: True)."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=250,  # full batch method normally
                        help="batch size")

    parser.add_argument("--batch-size-calculation",
                        type=non_negative_int,
                        default=250,
                        help="batch size used for calculating the gradient")

    parser.add_argument("--dropout",
                        action='store_true',
                        default=False,  # original didn't use dropout
                        help="Use dropout.")

    default_max_norm = 1.9365  # value used in original

    parser.add_argument("--max-filter-norm",
                        type=max_norm_arg,
                        default=default_max_norm,
                        help=("Max. L2 norm of the convolutional filters. "
                              "Enter a negative number to not impose any "
                              "max norm."))

    parser.add_argument("--max-col-norm",
                        type=max_norm_arg,
                        default=default_max_norm,
                        help=("Max. L2 norm of weight matrix columns. Enter a "
                              "negative number to not imppose any max norm."))

    parser.add_argument("--weight-decay",
                        type=non_negative_float,
                        default=0.00005,
                        metavar="K",
                        help=("For each weight matrix or filters tensor W, "
                              "add K * sqr(W).sum() to each batch's cost, for "
                              "all weights and filters"))

    parser.add_argument("--validation-size",
                        type=non_negative_int,
                        default=10000,
                        metavar="V",
                        help=("If this is zero, use the test set as the "
                              "validation set. Otherwise, use the last V "
                              "elements of the training set as the validation "
                              "set."))

    parser.add_argument("--armijo",
                        default=False,
                        action="store_true",
                        help="use Armijo line search procedure.")

    parser.add_argument("--tangent",
                        default=False,
                        action="store_true",
                        help="Use weight norm regularization by subtracting off the non-tangent component of the weight updates")

    parser.add_argument("--method",
                        default='FR',
                        type=correct_method,
                        help=("Choose a variant of the Conjugate Gradient algorithm: "
                              "'FR': Fletcher-Reeves, 'PB': Polak-Ribiere, 'HS': Hestenes-Stiefel or 'DY': Dai-Yuan."))

    return parser.parse_args()

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


def build_conv_classifier(input_node,
                          filter_shapes,
                          filter_counts,
                          filter_init_uniform_ranges,
                          pool_shapes,
                          pool_strides,
                          affine_output_sizes,
                          affine_init_stddevs,
                          dropout_include_rates,
                          rng,
                          theano_rng):
    '''
    Builds a classification convnet on top of input_node.

    Returns
    -------
    rval: tuple
      (conv_nodes, affine_nodes, output_node), where:
         conv_nodes is a list of the Conv2d nodes.
         affine_nodes is a list of the AffineNodes.
         output_node is the final node, a Softmax.
    '''

    assert_is_instance(input_node, RescaleImage)

    conv_shape_args = (filter_shapes,
                       pool_shapes,
                       pool_strides)

    for conv_shapes in conv_shape_args:
        for conv_shape in conv_shapes:
            assert_all_integer(conv_shape)
            assert_all_greater(conv_shape, 0)

    conv_args = conv_shape_args + (filter_counts, filter_init_uniform_ranges)
    assert_all_equal([len(c) for c in conv_args])

    assert_equal(len(affine_output_sizes), len(affine_init_stddevs))

    assert_equal(len(dropout_include_rates),
                 len(filter_shapes) + len(affine_output_sizes))

    assert_equal(affine_output_sizes[-1], 10)  # for MNIST

    assert_equal(input_node.output_format.axes, ('b', '0', '1'))

    #
    # Done sanity-checking args.
    #

    input_shape = input_node.output_format.shape

    # Converts from MNIST's ('b', '0', '1') to ('b', 'c', '0', '1')
    last_node = FormatNode(input_node,
                           DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(input_shape[0],
                                              1,
                                              input_shape[1],
                                              input_shape[2]),
                                       dtype=None),
                           {'b': ('b', 'c')})
        # {'1': ('1', 'c')})

    conv_dropout_include_rates = \
        dropout_include_rates[:len(filter_shapes)]

    # Adds a dropout-conv-bias-relu-maxpool stack for each element in
    # filter_XXXX

    conv_layers = []

    def uniform_init(rng, params, init_range):
        '''
        Fills params with values uniformly sampled from
        [-init_range, init_range]
        '''

        assert_floating(init_range)
        assert_greater_equal(init_range, 0)

        values = params.get_value()
        values[...] = rng.uniform(low=-init_range,
                                  high=init_range,
                                  size=values.shape)
        params.set_value(values)

    for (filter_shape,
         filter_count,
         filter_init_range,
         pool_shape,
         pool_stride,
         conv_dropout_include_rate) in safe_izip(filter_shapes,
                                                 filter_counts,
                                                 filter_init_uniform_ranges,
                                                 pool_shapes,
                                                 pool_strides,
                                                 conv_dropout_include_rates):
        if conv_dropout_include_rate != 1.0:
            last_node = Dropout(last_node,
                                conv_dropout_include_rate,
                                theano_rng)

        last_node = Conv2dLayer(last_node,
                                filter_shape,
                                filter_count,
                                conv_pads='valid',
                                pool_window_shape=pool_shape,
                                pool_strides=pool_stride,
                                pool_pads='pylearn2')
        conv_layers.append(last_node)

        uniform_init(rng, last_node.conv2d_node.filters, filter_init_range)

    affine_dropout_include_rates = dropout_include_rates[len(filter_shapes):]

    affine_layers = []

    def normal_distribution_init(rng, params, stddev):
        '''
        Fills params with values uniformly sampled from
        [-init_range, init_range]
        '''

        assert_floating(stddev)
        assert_greater_equal(stddev, 0)

        values = params.get_value()
        values[...] = rng.standard_normal(values.shape) * stddev
        params.set_value(values)

    #
    # Adds a dropout-affine-relu stack for each element in affine_XXXX,
    # except for the last one, where it omits the dropout.
    #

    for (affine_size,
         affine_init_stddev,
         affine_dropout_include_rate) in \
        safe_izip(affine_output_sizes,
                  affine_init_stddevs,
                  affine_dropout_include_rates):

        if affine_dropout_include_rate < 1.0:
            last_node = Dropout(last_node,
                                affine_dropout_include_rate,
                                theano_rng)

        # No need to supply an axis map for the first affine transform.
        # By default, it collapses all non-'b' axes into a feature vector,
        # which is what we want.

        # remap from bc01 to b01c before flattening to bf, as pylearn2 does,
        # just so that they do identical things.
        last_node = SoftmaxLayer(last_node,
                                 DenseFormat(axes=('b', 'f'),
                                             shape=(-1, affine_size),
                                             dtype=None),
                                 input_to_bf_map={('0', '1', 'c'): 'f'})
        normal_distribution_init(rng,
                                 last_node.affine_node.linear_node.params,
                                 affine_init_stddev)
        # stddev_init(rng, last_node.bias_node.params, affine_init_stddev)
        affine_layers.append(last_node)

    #################################################################################################
    ### BUILD THE SECOND NETWORK WITH FLAT PARAMETERS (given the dimensions of the first) ###########
    #################################################################################################

    rng = numpy.random.RandomState(281934)
    std_deviation = .05

    # Fetch all parameters and shapes
    parameters = []
    for conv_layer in conv_layers:
        filters = conv_layer.conv2d_node.filters
        parameters.append(filters)

        bias = conv_layer.bias_node.params
        parameters.append(bias)

    for affine_layer in affine_layers:
        weights = affine_layer.affine_node.linear_node.params
        parameters.append(weights)

        biases = affine_layer.affine_node.bias_node.params
        parameters.append(biases)

    '''
    print(len(parameters))
    for parameter in parameters:
        print(parameter.get_value().shape)

    shapes = []
    params_flat_values = numpy.asarray([], dtype=theano.config.floatX)
    counter = 0
    for parameter in parameters:
        shape = parameter.get_value().shape
        if counter%2 == 0 and len(shape)==4:
            vector_param = numpy.asarray(numpy.ndarray.flatten(parameter.get_value()), dtype=theano.config.floatX)
            vector_param[...] = rng.standard_normal(vector_param.shape) * std_deviation
            col_length = shape[2]
            index_from = 0

            ###
            #for _ in range(shape[0]*shape[1]*shape[3]):
            #    index_to = index_from + col_length
            #    vector_param[index_from:index_to] = vector_param[index_from:index_to]/numpy.linalg.norm(vector_param[index_from:index_to])
            #    index_from = index_to
            ####

        elif counter%2==0:
            vector_param = numpy.asarray(numpy.ndarray.flatten(parameter.get_value()), dtype=theano.config.floatX)
            vector_param[...] = rng.standard_normal(vector_param.shape) * std_deviation
        else:
            vector_param = numpy.asarray(numpy.ndarray.flatten(parameter.get_value()), dtype=theano.config.floatX)

        params_flat_values = numpy.append(params_flat_values, vector_param)
        shapes.append(shape)
    '''
    params_flat_values = numpy.asarray([], dtype=theano.config.floatX)
    shapes = []

    for parameter in parameters:
        parameter_value = parameter.get_value()
        shapes.append(parameter_value.shape)
        vector_param = numpy.asarray(numpy.ndarray.flatten(parameter_value))
        params_flat_values = numpy.append(params_flat_values, vector_param)

    print(params_flat_values)
    print(params_flat_values.shape)

    params_flat = theano.shared(params_flat_values)
    params_old_flat = theano.shared(params_flat_values)

    '''
    Builds a classification convnet on top of input_node.

    Returns
    -------
    rval: tuple
      (conv_nodes, affine_nodes, output_node), where:
         conv_nodes is a list of the Conv2d nodes.
         affine_nodes is a list of the AffineNodes.
         output_node is the final node, a Softmax.
    '''

    assert_is_instance(input_node, RescaleImage)

    conv_shape_args = (filter_shapes,
                       pool_shapes,
                       pool_strides)

    for conv_shapes in conv_shape_args:
        for conv_shape in conv_shapes:
            assert_all_integer(conv_shape)
            assert_all_greater(conv_shape, 0)

    conv_args = conv_shape_args + (filter_counts, filter_init_uniform_ranges)
    assert_all_equal([len(c) for c in conv_args])

    assert_equal(len(affine_output_sizes), len(affine_init_stddevs))

    assert_equal(len(dropout_include_rates),
                 len(filter_shapes) + len(affine_output_sizes))

    assert_equal(affine_output_sizes[-1], 10)  # for MNIST

    assert_equal(input_node.output_format.axes, ('b', '0', '1'))

    #
    # Done sanity-checking args.
    #

    input_shape = input_node.output_format.shape

    # Converts from MNIST's ('b', '0', '1') to ('b', 'c', '0', '1')
    last_node = FormatNode(input_node,
                           DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(input_shape[0],
                                              1,
                                              input_shape[1],
                                              input_shape[2]),
                                       dtype=None),
                           {'b': ('b', 'c')})
        # {'1': ('1', 'c')})

    conv_dropout_include_rates = \
        dropout_include_rates[:len(filter_shapes)]

    # Adds a dropout-conv-bias-relu-maxpool stack for each element in
    # filter_XXXX

    conv_layers = []
    counter = 0
    index_from = 0

    for (filter_shape,
         filter_count,
         filter_init_range,
         pool_shape,
         pool_stride,
         conv_dropout_include_rate) in safe_izip(filter_shapes,
                                                 filter_counts,
                                                 filter_init_uniform_ranges,
                                                 pool_shapes,
                                                 pool_strides,
                                                 conv_dropout_include_rates):
        if conv_dropout_include_rate != 1.0:
            last_node = Dropout(last_node,
                                conv_dropout_include_rate,
                                theano_rng)

        print(shapes)

        shape1 = shapes[counter]
        shape2 = shapes[counter+1]
        size1= numpy.prod(numpy.asarray(shape1))
        size2= numpy.prod(numpy.asarray(shape2))
        index_to = index_from + size1
        #filters_ = theano.tensor.transpose(params_flat[index_from:index_to].reshape(shape1), axes=[0,1,3,2])
        filters_ = params_flat[index_from:index_to].reshape(shape1)
        index_from = index_to
        index_to = index_from + size2
        bias_ = params_flat[index_from:index_to].reshape(shape2)
        index_from = index_to

        last_node = Conv2dLayer(last_node,
                                filter_shape,
                                filter_count,
                                conv_pads='valid',
                                pool_window_shape=pool_shape,
                                pool_strides=pool_stride,
                                pool_pads='pylearn2',
                                filters=filters_,
                                bias=bias_)
        conv_layers.append(last_node)

        counter = counter + 2


    affine_dropout_include_rates = dropout_include_rates[len(filter_shapes):]

    affine_layers = []

    #
    # Adds a dropout-affine-relu stack for each element in affine_XXXX,
    # except for the last one, where it omits the dropout.
    #

    for (affine_size,
         affine_init_stddev,
         affine_dropout_include_rate) in \
        safe_izip(affine_output_sizes,
                  affine_init_stddevs,
                  affine_dropout_include_rates):

        if affine_dropout_include_rate < 1.0:
            last_node = Dropout(last_node,
                                affine_dropout_include_rate,
                                theano_rng)

        # No need to supply an axis map for the first affine transform.
        # By default, it collapses all non-'b' axes into a feature vector,
        # which is what we want.

        shape1 = shapes[counter]
        #shape1 = (shape1[1], shape1[0])
        shape2 = shapes[counter+1]
        size1= numpy.prod(numpy.asarray(shape1))
        size2= numpy.prod(numpy.asarray(shape2))
        index_to = index_from + size1
        weights_ = params_flat[index_from:index_to].reshape(shape1)
        index_from = index_to
        index_to = index_from + size2
        bias_ = params_flat[index_from:index_to].reshape(shape2)
        index_from = index_to
        # remap from bc01 to b01c before flattening to bf, as pylearn2 does,
        # just so that they do identical things.
        last_node = SoftmaxLayer(last_node,
                                 DenseFormat(axes=('b', 'f'),
                                             shape=(-1, affine_size),
                                             dtype=None),
                                 weights=weights_,
                                 bias=bias_,
                                 input_to_bf_map={('0', '1', 'c'): 'f'})

        # stddev_init(rng, last_node.bias_node.params, affine_init_stddev)
        affine_layers.append(last_node)


    return conv_layers, affine_layers, last_node, params_flat, params_old_flat, shapes


def print_misclassification_rate(values, _):  # ignores 2nd argument (formats)
    '''
    Prints the misclassification rate.
    '''
    print("Misclassification rate: %s" % str(values))


def print_loss(values, _):  # ignores 2nd argument (formats)
    '''
    Prints the average loss.
    '''
    print("Average loss: %s" % str(values))


def main():
    '''
    Entry point of this script.
    '''

    args = parse_args()

    # Hyperparameter values taken from Pylearn2:
    # In pylearn2/scripts/tutorials/convolutional_network/:
    #   convolutional_network.ipynb

    filter_counts = [64, 64]
    filter_init_uniform_ranges = [.05] * len(filter_counts)
    filter_shapes = [(5, 5), (5, 5)]
    pool_shapes = [(4, 4), (4, 4)]
    pool_strides = [(2, 2), (2, 2)]
    affine_output_sizes = [10]
    affine_init_stddevs = [.05] * len(affine_output_sizes)
    dropout_include_rates = ([.5 if args.dropout else 1.0] *
                             (len(filter_counts) + len(affine_output_sizes)))

    assert_equal(affine_output_sizes[-1], 10)

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

    image_node = RescaleImage(image_lookup_node)

    rng = numpy.random.RandomState(129734)
    theano_rng = RandomStreams(2387845)

    (conv_layers,
     affine_layers,
     output_node,
     params_flat,
     params_old_flat,
     shapes) = build_conv_classifier(image_node,
                                          filter_shapes,
                                          filter_counts,
                                          filter_init_uniform_ranges,
                                          pool_shapes,
                                          pool_strides,
                                          affine_output_sizes,
                                          affine_init_stddevs,
                                          dropout_include_rates,
                                          rng,
                                          theano_rng)

    loss_node = CrossEntropy(output_node, label_lookup_node)
    scalar_loss = loss_node.output_symbol.mean()
#    scalar_loss2 = theano.clone(scalar_loss, replace = {params_flat: params_old_flat})

    if args.weight_decay != 0.0:
        for conv_layer in conv_layers:
            filters = conv_layer.conv2d_node.filters
            filter_loss = args.weight_decay * theano.tensor.sqr(filters).sum()
            scalar_loss = scalar_loss + filter_loss

        for affine_layer in affine_layers:
            weights = affine_layer.affine_node.linear_node.params
            weight_loss = args.weight_decay * theano.tensor.sqr(weights).sum()
            scalar_loss = scalar_loss + weight_loss

    max_epochs = 200

    #
    # Makes parameter updater
    #

    gradient = theano.gradient.grad(scalar_loss, params_flat)

    loss_function = theano.function([input_indices_symbolic.output_symbol],scalar_loss)
    gradient_function = theano.function([input_indices_symbolic.output_symbol],gradient)

    cost_arguments = mnist_training_iterator.next()
    print(loss_function(*cost_arguments))
    grads = gradient_function(*cost_arguments)
    print(grads)
    print(grads.shape)

    #
    # Makes batch and epoch callbacks
    #

    '''
    def make_output_filename(args, best=False):
            
            Constructs a filename that reflects the command-line params.
            
            assert_equal(os.path.splitext(args.output_prefix)[1], "")

            if os.path.isdir(args.output_prefix):
                output_dir, output_prefix = args.output_prefix, ""
            else:
                output_dir, output_prefix = os.path.split(args.output_prefix)
                assert_true(os.path.isdir(output_dir))

            if output_prefix != "":
                output_prefix = output_prefix + "_"

            output_prefix = os.path.join(output_dir, output_prefix)

            return ("%slr-%g_mom-%g_nesterov-%s_bs-%d%s.pkl" %
                    (output_prefix,
                     args.learning_rate,
                     args.initial_momentum,
                     not args.no_nesterov,
                     args.batch_size,
                     "_best" if best else ""))
    '''


    # Set up the loggers

    assert_equal(os.path.splitext(args.output_prefix)[1], "")
    if os.path.isdir(args.output_prefix) and \
       not args.output_prefix.endswith('/'):
        args.output_prefix += '/'

    output_dir, output_prefix = os.path.split(args.output_prefix)
    if output_prefix != "":
        output_prefix = output_prefix + "_"

    output_prefix = os.path.join(output_dir, output_prefix)

    epoch_logger = EpochLogger(output_prefix + "CG_mini_batch_" + str(args.batch_size) + ".h5")


    misclassification_node = Misclassification(output_node, label_lookup_node)

    validation_loss_monitor = MeanOverEpoch(loss_node, callbacks=[])
    epoch_logger.subscribe_to('validation mean loss', validation_loss_monitor)

    training_stopper = StopsOnStagnation(max_epochs=20,
                                             min_proportional_decrease=0.0)
    validation_misclassification_monitor = MeanOverEpoch(misclassification_node,
                                             callbacks=[print_misclassification_rate,
                                                        training_stopper])

    epoch_logger.subscribe_to('validation misclassification',
                                validation_misclassification_monitor)

    # batch callback (monitor)
    #training_loss_logger = LogsToLists()
    training_loss_monitor = MeanOverEpoch(loss_node,
                                          callbacks=[print_loss])
    epoch_logger.subscribe_to("training loss", training_loss_monitor)

    training_misclassification_monitor = MeanOverEpoch(misclassification_node,
                                                       callbacks=[])
    epoch_logger.subscribe_to('training misclassification %',
                              training_misclassification_monitor)


    #model = SerializableModel([input_indices_symbolic], [output_node])
    #saves_best = SavesAtMinimum(model, make_output_filename(args, best=True))

    validation_loss_monitor = MeanOverEpoch(loss_node,
                                            callbacks=[])
    epoch_logger.subscribe_to("Validation Loss", validation_loss_monitor)

    epoch_timer = EpochTimer2()
    epoch_logger.subscribe_to('epoch duration', epoch_timer)

    validation_callback = ValidationCallback(
        inputs=[input_indices_symbolic.output_symbol],
        input_iterator=mnist_validation_iterator,
        epoch_callbacks=[validation_loss_monitor,
                         validation_misclassification_monitor])

    # trainer = Sgd((image_node.output_symbol, label_node.output_symbol),
    trainer = CG(inputs=[input_indices_symbolic],
                  parameters=params_flat,
                  gradient=gradient,
                  learning_rate=args.learning_rate,
                  training_iterator=mnist_training_iterator,
                  validation_iterator=mnist_validation_iterator,
                  scalar_loss=scalar_loss,
                  armijo=args.armijo,
                  tangent=args.tangent,
                  method=args.method,
                  batch_size=args.batch_size,
                  epoch_callbacks=([
                             #training_loss_monitor,
                             # training_misclassification_monitor,
                              validation_callback,
                              LimitsNumEpochs(max_epochs),
                              epoch_timer]),
                   param_shapes=shapes)

    '''
    stuff_to_pickle = OrderedDict(
        (('model', model),
         ('validation_loss_logger', validation_loss_logger)))

    # Pickling the trainer doesn't work when there are Dropout nodes.
    # stuff_to_pickle = OrderedDict(
    #     (('trainer', trainer),
    #      ('validation_loss_logger', validation_loss_logger),
    #      ('model', model)))

    trainer.epoch_callbacks += (momentum_updaters +
                                [EpochTimer(),
                                 PicklesOnEpoch(stuff_to_pickle,
                                                make_output_filename(args),
                                                overwrite=False),
                                 validation_callback,
                                 LimitsNumEpochs(max_epochs)])
    '''

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    print("Total elapsed time is for training is: ", elapsed_time)


if __name__ == '__main__':
    main()
