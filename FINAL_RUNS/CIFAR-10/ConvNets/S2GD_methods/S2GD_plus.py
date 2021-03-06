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
                               Node,
                               Lcn,
                               AffineLayer)
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
from extension_semi_stochastic import SemiSgd, SemiSgdParameterUpdater

class ImageLookeupNode(Node):

    def __init__(self,input_node, images_array):

        self.images = images_array
        output_symbol = self.images[input_node.output_symbol]
        output_format = DenseFormat(axes=('b','c', '0', '1'),
                                    shape=(-1,3, 32, 32),
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
                                    dtype='int64')

        super(LabelLookeupNode, self).__init__(input_nodes=input_node,
                                        output_symbol=output_symbol,
                                        output_format=output_format)

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
        if arg != 'SGD' and arg != 'S2GD' and arg != 'S2GD_plus' and arg != 'S2GD_rolling':
            raise ValueError("Please enter a valid choice for optimization method: 'SGD', 'S2GD', 'S2GD_plus' or 'S2GD_rolling'"
                             % arg)

        return arg

    parser.add_argument("--output-prefix",
                        type=legit_prefix,
                        required=False,
			default='output',
                        help=("Directory and optional prefix of filename to "
                              "save the log to."))

    # The default hyperparameter values below are taken from Pylearn2's mlp
    # tutorial, in pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml
    #
    # Exceptions were made, since the original hyperparameters led to
    # divergence. These changes have been marked below.

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=0.05,  # .01 used in pylearn2 demo
                        help=("Learning rate."))

    parser.add_argument("--initial-momentum",
                        type=non_negative_0_to_1,
                        default=0.0,  # 0.5 used in original
                        help=("Initial momentum."))

    parser.add_argument("--nesterov",
                        default=False,  # original didn't use nesterov
                        action="store_true",
                        help=("Don't use Nesterov accelerated gradients "
                              "(default: False)."))

    parser.add_argument("--no-shuffle-dataset",
                        default=False,  # original didn't use nesterov
                        action="store_true",
                        help=("Shuffle dataset before running "
                              "(default: True)."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=125,
                        help="batch size")

    parser.add_argument("--batch-size-full",
                        type=non_negative_int,
                        default=100,
                        help="batch size for calculating the full gradient")

    parser.add_argument("--dropout",
                        action='store_true',
                        default=True,  # original didn't use dropout
                        help="Use dropout.")

    parser.add_argument("--final-momentum",
                        type=non_negative_float,
                        default=0.0,  # original used .99
                        help="Value for momentum to linearly scale up to.")

    parser.add_argument("--epochs-to-momentum-saturation",
                        default=10,
                        type=positive_int,
                        help=("# of epochs until momentum linearly scales up "
                              "to --momentum_final_value."))

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

    parser.add_argument("--method",
                        default='S2GD_plus',
                        type=correct_method,
                        help=("Choose method: 'SGD', 'S2GD', 'S2GD_plus', or 'S2GD_rolling'"))

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


def build_conv_classifier(input_node,
                          filter_shapes,
                          filter_counts,
                          filter_init_uniform_ranges,
                          pool_shapes,
                          pool_strides,
                          affine_output_sizes,
                          affine_init_stddevs,
                          dropout_include_rates,
                          conv_pads,
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

    assert_is_instance(input_node, Lcn)

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

    #assert_equal(input_node.output_format.axes, ('b', '0', '1'))

    #
    # Done sanity-checking args.
    #

    input_shape = input_node.output_format.shape

    # Converts from MNIST's ('b', '0', '1') to ('b', 'c', '0', '1')
    last_node = input_node

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
         conv_dropout_include_rate,
         conv_pads)                 in safe_izip(filter_shapes,
                                                 filter_counts,
                                                 filter_init_uniform_ranges,
                                                 pool_shapes,
                                                 pool_strides,
                                                 conv_dropout_include_rates,
                                                 conv_pads):
        if conv_dropout_include_rate != 1.0:
            last_node = Dropout(last_node,
                                conv_dropout_include_rate,
                                theano_rng)

        last_node = Conv2dLayer(last_node,
                                filter_shape,
                                filter_count,
                                conv_pads=conv_pads,
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

    # Add a fully connected layer here:
    output_format = DenseFormat(axes=('b', 'f'),
                                    shape=(-1, 500),
                                    dtype=None)

    if affine_dropout_include_rates < 1.0:
        last_node = Dropout(last_node,
                            affine_dropout_include_rates,
                            theano_rng)

    last_node = AffineLayer(last_node, output_format, input_to_bf_map={('0', '1', 'c'): 'f'})
    affine_layers.append(last_node)
    normal_distribution_init(rng,
                         last_node.affine_node.linear_node.params,
                         0.05)


    for (affine_size,
         affine_init_stddev,
         affine_dropout_include_rate) in \
        safe_izip(affine_output_sizes,
                  affine_init_stddevs,
                  affine_dropout_include_rates):

        '''
        if affine_dropout_include_rate < 1.0:
            last_node = Dropout(last_node,
                                affine_dropout_include_rate,
                                theano_rng)
        '''
        # No need to supply an axis map for the first affine transform.
        # By default, it collapses all non-'b' axes into a feature vector,
        # which is what we want.

        # remap from bc01 to b01c before flattening to bf, as pylearn2 does,
        # just so that they do identical things.
        last_node = SoftmaxLayer(last_node,
                                 DenseFormat(axes=('b', 'f'),
                                             shape=(-1, affine_size),
                                             dtype=None))
                                 #input_to_bf_map={('0', '1', 'c'): 'f'})
        normal_distribution_init(rng,
                                 last_node.affine_node.linear_node.params,
                                 affine_init_stddev)
        # stddev_init(rng, last_node.bias_node.params, affine_init_stddev)
        affine_layers.append(last_node)

    return conv_layers, affine_layers, last_node


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

    filter_counts = [96, 192, 192]
    filter_init_uniform_ranges = [0.005]* len(filter_counts)
    filter_shapes = [(8, 8), (8,8), (5, 5)]
    pool_shapes = [(4, 4),(4, 4), (2, 2)]
    pool_strides = [(2, 2), (2, 2), (2,2)]
    pool_pads = [(2,2), (2,2), (2,2)]
    affine_output_sizes = [10]
    affine_init_stddevs = [.05] * len(affine_output_sizes)
    dropout_include_rates = [0.8, 0.5, 0.5, 0.5]
    #dropout_include_rates = ([.8 if args.dropout else 1.0] *
    #                         (len(filter_counts) + len(affine_output_sizes)))
    conv_pads = [(4, 4), (3, 3), (3, 3)]

    assert_equal(affine_output_sizes[-1], 10)

    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    batch1 = unpickle('/home/s1422538/datasets/simplelearn/cifar10/original_files/cifar-10-batches-py/data_batch_1')
    batch2 = unpickle('/home/s1422538/datasets/simplelearn/cifar10/original_files/cifar-10-batches-py/data_batch_2')
    batch3 = unpickle('/home/s1422538/datasets/simplelearn/cifar10/original_files/cifar-10-batches-py/data_batch_3')
    batch4 = unpickle('/home/s1422538/datasets/simplelearn/cifar10/original_files/cifar-10-batches-py/data_batch_4')
    batch5 = unpickle('/home/s1422538/datasets/simplelearn/cifar10/original_files/cifar-10-batches-py/data_batch_5')

    training_tensors = [ numpy.concatenate((batch1['data'].reshape(10000,3,32,32), batch2['data'].reshape(10000,3,32,32), batch3['data'].reshape(10000,3,32,32), batch4['data'].reshape(10000,3,32,32) )), numpy.concatenate((batch1['labels'], batch2['labels'], batch3['labels'], batch4['labels'])) ]
    validation_tensors = [ batch5['data'].reshape(10000,3,32,32), numpy.asarray(batch5['labels']) ]

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
    indices_training_iterator = indices_training_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size)
    indices_validation_iterator = indices_validation_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size)
    training_iterator_full = indices_training_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size_full)

    mnist_validation_iterator = indices_validation_iterator
    mnist_training_iterator = indices_training_iterator

    input_indices_symbolic, = indices_training_iterator.make_input_nodes()
    image_lookup_node = ImageLookeupNode(input_indices_symbolic, all_images_shared)
    label_lookup_node = LabelLookeupNode(input_indices_symbolic, all_labels_shared)

    image_node = RescaleImage(image_lookup_node)
    image_node = Lcn(image_node)

    rng = numpy.random.RandomState(129734)
    theano_rng = RandomStreams(2387845)

    (conv_layers,
     affine_layers,
     output_node) = build_conv_classifier(image_node,
                                          filter_shapes,
                                          filter_counts,
                                          filter_init_uniform_ranges,
                                          pool_shapes,
                                          pool_strides,
                                          affine_output_sizes,
                                          affine_init_stddevs,
                                          dropout_include_rates,
                                          conv_pads,
                                          rng,
                                          theano_rng)

    loss_node = CrossEntropy(output_node, label_lookup_node)
    scalar_loss = loss_node.output_symbol.mean()

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
    # Extract variables
    #

    parameters = []
    old_parameters = []

    def add_updaters(parameter,
                     old_parameter,
                     scalar_loss,
                     scalar_loss_at_old_params,
                     parameter_updaters,
                     momentum_updaters):
        '''
        Adds a ParameterUpdater to parameter_updaters, and a
        LinearlyInterpolatesOverEpochs to momentum_updaters.
        '''
        gradient = theano.gradient.grad(scalar_loss, parameter)
        all_gradients.append(gradient)
        gradient_at_old_params = theano.gradient.grad(scalar_loss_at_old_params, old_parameter)
        parameter_updaters.append(SemiSgdParameterUpdater(parameter,
                                                          gradient,
                                                          gradient_at_old_params,
                                                          args.learning_rate,
                                                          args.initial_momentum,
                                                          args.method,
                                                          mnist_training_iterator,
                                                          training_iterator_full,
                                                          args.nesterov))
        momentum_updaters.append(LinearlyInterpolatesOverEpochs(
            parameter_updaters[-1].momentum,
            args.final_momentum,
            args.epochs_to_momentum_saturation))

    for conv_layer in conv_layers:
        filters = conv_layer.conv2d_node.filters
        parameters.append(filters)
        old_param = theano.shared(numpy.zeros(filters.get_value().shape, dtype=filters.dtype))
        old_parameters.append(old_param)

        bias = conv_layer.bias_node.params
        parameters.append(bias)
        old_param = theano.shared(numpy.zeros(bias.get_value().shape, dtype=bias.dtype))
        old_parameters.append(old_param)

    for affine_layer in affine_layers:
        weights = affine_layer.affine_node.linear_node.params
        parameters.append(weights)
        old_param = theano.shared(numpy.zeros(weights.get_value().shape, dtype=weights.dtype))
        old_parameters.append(old_param)

        biases = affine_layer.affine_node.bias_node.params
        parameters.append(biases)
        old_param = theano.shared(numpy.zeros(biases.get_value().shape, dtype=biases.dtype))
        old_parameters.append(old_param)


#    loss_node2 = theano.clone(loss_node, replace = {parameter: old_parameter for parameter,old_parameter in safe_izip(parameters, old_parameters)} )
    scalar_loss2 = theano.clone(scalar_loss, replace = {parameter: old_parameter for parameter,old_parameter in safe_izip(parameters, old_parameters)} )
#   scalar_loss2 = loss_node2.output_symbol.mean()

    # Create the parameters updaters
    parameter_updaters = []
    momentum_updaters = []
    all_gradients = []
    counter = 0

    for conv_layer in conv_layers:
        filters = conv_layer.conv2d_node.filters
        old_params = old_parameters[counter]

        add_updaters(filters,
                     old_params,
                     scalar_loss,
                     scalar_loss2,
                     parameter_updaters,
                     momentum_updaters)

        counter = counter + 1

        if args.max_filter_norm != numpy.inf:
            limit_param_norms(parameter_updaters[-1],
                              filters,
                              args.max_filter_norm,
                              (1, 2, 3))

        bias = conv_layer.bias_node.params
        old_params = old_parameters[counter]

        add_updaters(bias,
                     old_params,
                     scalar_loss,
                     scalar_loss2,
                     parameter_updaters,
                     momentum_updaters)

        counter = counter + 1

    for affine_layer in affine_layers:
        weights = affine_layer.affine_node.linear_node.params
        old_params = old_parameters[counter]

        add_updaters(weights,
                     old_params,
                     scalar_loss,
                     scalar_loss2,
                     parameter_updaters,
                     momentum_updaters)

        counter = counter + 1

        if args.max_col_norm != numpy.inf:
            limit_param_norms(parameter_updater=parameter_updaters[-1],
                              param=weights,
                              max_norm=args.max_col_norm,
                              input_axes=[0])

        biases = affine_layer.affine_node.bias_node.params
        old_params = old_parameters[counter]

        add_updaters(biases,
                     old_params,
                     scalar_loss,
                     scalar_loss2,
                     parameter_updaters,
                     momentum_updaters)

        counter = counter + 1

    '''
    print(parameters)
    print(len(parameters))

    for param in parameters:
        print(param.get_value().shape)
    '''

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
                     args.nesterov,
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

    epoch_logger = EpochLogger(output_prefix + "S2GD_plus.h5")


    misclassification_node = Misclassification(output_node, label_lookup_node)

    validation_loss_monitor = MeanOverEpoch(loss_node, callbacks=[])
    epoch_logger.subscribe_to('validation mean loss', validation_loss_monitor)

    training_stopper = StopsOnStagnation(max_epochs=10,
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

    epoch_timer = EpochTimer2()
    epoch_logger.subscribe_to('epoch duration', epoch_timer)
#    epoch_logger.subscribe_to('epoch time',
 #                             epoch_timer)
    #################


    #model = SerializableModel([input_indices_symbolic], [output_node])
    #saves_best = SavesAtMinimum(model, make_output_filename(args, best=True))

    validation_loss_monitor = MeanOverEpoch(loss_node,
                                            callbacks=[])
    epoch_logger.subscribe_to("Validation Loss", validation_loss_monitor)

    validation_callback = ValidationCallback(
        inputs=[input_indices_symbolic.output_symbol],
        input_iterator=mnist_validation_iterator,
        epoch_callbacks=[validation_loss_monitor,
                         validation_misclassification_monitor])

    loss_function = theano.function([input_indices_symbolic.output_symbol], scalar_loss)
    cost_args = mnist_validation_iterator.next()
    print(loss_function(*cost_args))

    # trainer = Sgd((image_node.output_symbol, label_node.output_symbol),
    trainer = SemiSgd([input_indices_symbolic],
                    mnist_training_iterator,
                    parameters,
                    old_parameters,
                    parameter_updaters,
                    training_iterator_full,
                    epoch_callbacks=(parameter_updaters +
                             momentum_updaters +
                             [#training_loss_monitor,
                              #training_misclassification_monitor,
                              validation_callback,
                              LimitsNumEpochs(max_epochs),
                              epoch_timer]))

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

    print(args.method, " is now executed")
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    print("Total elapsed time is for training is: ", elapsed_time)


if __name__ == '__main__':
    main()
