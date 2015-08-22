#! /usr/bin/env python

import os
import sys
sys.dont_write_bytecode = True
import argparse
import numpy
import theano
import time
import timeit
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
                               SoftmaxLayer)
from simplelearn.utils import safe_izip
from simplelearn.asserts import (assert_all_greater,
                                 assert_all_less_equal,
                                 assert_all_integer)
from simplelearn.io import SerializableModel
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (Sgd,
                                  # LogsToLists,
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
from RMSprop_extensions import RMSpropNesterovSgdParameterUpdater

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
                        default=0.01,
                        help=("Learning rate."))

    parser.add_argument("--initial-momentum",
                        type=non_negative_float,
                        default=0.5,
                        help=("Initial momentum."))

    parser.add_argument("--nesterov",
                        default=True,
                        action="store_true",
                        help=("Use Nesterov accelerated gradients (default: "
                              "False)."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=100,
                        help="batch size")

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

    parser.add_argument("--final-momentum",
                        type=positive_0_to_1,
                        default=.5,  # .99 used in pylearn2 demo
                        help="Value for momentum to linearly scale up to.")

    parser.add_argument("--epochs-to-momentum-saturation",
                        default=10,
                        type=positive_int,
                        help=("# of epochs until momentum linearly scales up "
                              "to --momentum_final_value."))

    parser.add_argument("--validation-size",
                        type=non_negative_int,
                        default=10000,
                        metavar="V",
                        help=("Hold out the last V examples of the training "
                              "set to use as a validation set. If V is 0, "
                              "use the MNIST test set as the validation set."))

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
        '''
        Mimics the sparse initialization in
        pylearn2.models.mlp.Linear.set_input_space()
        '''

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

    return affine_nodes, last_node


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
    indices_training_iterator = indices_training_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size)
    indices_validation_iterator = indices_validation_dataset.iterator(iterator_type='sequential',batch_size=args.batch_size)

    mnist_validation_iterator = indices_validation_iterator
    mnist_training_iterator = indices_training_iterator

    input_indices_symbolic, = indices_training_iterator.make_input_nodes()
    image_lookup_node = ImageLookeupNode(input_indices_symbolic, all_images_shared)
    label_lookup_node = LabelLookeupNode(input_indices_symbolic, all_labels_shared)

    image_node = CastNode(image_lookup_node, 'floatX')
    # image_node = RescaleImage(image_uint8_node)

    rng = numpy.random.RandomState(34523)
    theano_rng = RandomStreams(23845)

    (affine_nodes,
     output_node) = build_fc_classifier(image_node,
                                        sizes,
                                        sparse_init_counts,
                                        args.dropout_include_rates,
                                        rng,
                                        theano_rng)

    loss_node = CrossEntropy(output_node, label_lookup_node)
    loss_sum = loss_node.output_symbol.mean()
    max_epochs = 200

    #
    # Makes parameter updaters
    #

    parameters = []
    parameter_updaters = []
    momentum_updaters = []
    for affine_node in affine_nodes:
        for params in (affine_node.linear_node.params,
                       affine_node.bias_node.params):
            parameters.append(params)
            gradients = theano.gradient.grad(loss_sum, params)
            parameter_updater = RMSpropNesterovSgdParameterUpdater(params,
                                                    gradients,
                                                    args.learning_rate,
                                                    args.initial_momentum,
                                                    args.nesterov)
            parameter_updaters.append(parameter_updater)

            momentum_updaters.append(LinearlyInterpolatesOverEpochs(
                parameter_updater.momentum,
                args.final_momentum,
                args.epochs_to_momentum_saturation))

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

    epoch_logger = EpochLogger(output_prefix + "SGD_RMSprop_nesterov.h5")

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

    epoch_timer = EpochTimer2()
    epoch_logger.subscribe_to('epoch duration', epoch_timer)

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

    validation_callback = ValidationCallback(
        inputs=[input_indices_symbolic.output_symbol],
        input_iterator=mnist_validation_iterator,
        epoch_callbacks=[validation_loss_monitor,
                         validation_misclassification_monitor])

    trainer = Sgd([input_indices_symbolic],
                  mnist_training_iterator,
                  callbacks=(parameter_updaters +
                             momentum_updaters +
                             [#training_loss_monitor,
                              #training_misclassification_monitor,
                              validation_callback,
                              LimitsNumEpochs(max_epochs),
                              epoch_timer]))
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

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    print("Total elapsed time is for training is: ", elapsed_time)


if __name__ == '__main__':
    main()
