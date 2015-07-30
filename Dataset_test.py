import os
import argparse
import numpy
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
                               SoftmaxLayer)
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
                                  # LogsToLists,
                                  SavesAtMinimum,
                                  MeanOverEpoch,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  # PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation,
                                  EpochLogger)
import pdb

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

mnist_training = Dataset(tensors=training_tensors,
                         names=mnist_training.names,
                         formats=mnist_training.formats)
mnist_validation = Dataset(tensors=validation_tensors,
                           names=mnist_training.names,
                           formats=mnist_training.formats)

mnist_validation_iterator = mnist_validation.iterator(
iterator_type='sequential',
batch_size=args.batch_size)
image_uint8_node, label_node = mnist_validation_iterator.make_input_nodes()
image_node = CastNode(image_uint8_node, 'floatX')
# image_node = RescaleImage(image_uint8_node)
