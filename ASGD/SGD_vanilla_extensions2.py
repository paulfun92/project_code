import os
import copy
import warnings
import cPickle
from collections import Sequence, OrderedDict
import h5py
import numpy
import theano
import theano.tensor as T
from nose.tools import (assert_true,
                        assert_is,
                        assert_equal,
                        assert_less,
                        assert_less_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance,
                        assert_is_not,
                        assert_in)
from simplelearn.asserts import (assert_integer,
                                 assert_floating,
                                 assert_all_less,
                                 assert_all_greater_equal,
                                 assert_all_integer,
                                 assert_is_subdtype,
                                 assert_all_is_instance,
                                 assert_parent_dir_exists)
from simplelearn.data import DataIterator
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.training import (LimitsNumEpochs, EpochCallback, StopTraining, ParameterUpdater, IterationCallback,
                                  EpochTimer2)
from simplelearn.data.dataset import DataIterator
from simplelearn.nodes import Node
import pdb



class Sgd(object):

    '''
    Uses stochastic gradient descent to optimize a cost w.r.t. parameters.

    The parameters and the inputs may be the same.

    At each iteration this computes the gradients of each parameter with
    respect to the cost function, then updates the parameter value using
    the gradients. How this update is performed (e.g. learning rate,
    momentum value & type, etc) is up to the SgdParameterUpdater
    objects passed into the constructor.
    '''

    def __init__(self,
                inputs,
                parameters,
                gradient,
                learning_rate,
                training_iterator,
                scalar_loss,
                batch_size,
                epoch_callbacks):


            #
            # sanity-checks the arguments.
            #

            assert_all_is_instance(inputs, Node)
            assert_is_instance(training_iterator, DataIterator)
            #assert_true(training_iterator.next_is_new_epoch())

            '''
            for (input,
                 iterator_input) in safe_izip(inputs,
                                              training_iterator.make_input_nodes()):
                assert_equal(input.output_format, iterator_input.output_format)
            '''

            assert_equal(len(epoch_callbacks),
                         len(frozenset(epoch_callbacks)),
                         "There were duplicate callbacks.")

            assert_all_is_instance(epoch_callbacks, EpochCallback)


            #
            # Sets members
            #

            self.parameters = parameters
            self.training_iterator = training_iterator
            self.learning_rate = learning_rate

            input_symbols = [i.output_symbol for i in inputs]

            self.epoch_callbacks = tuple(epoch_callbacks)

            self._train_called = False

            self.gradient_function = theano.function(input_symbols,gradient)
            self.loss_function = theano.function(input_symbols,scalar_loss)

            self.new_epoch = True
    #        self.method = self._parameter_updaters[0].method

            total_size_dataset = self.training_iterator.dataset.tensors[0].shape[0]
            self.batches_in_epoch = total_size_dataset / batch_size

            batch_size_for_calculation =self.training_iterator.batch_size
            assert_less_equal(batch_size_for_calculation, batch_size)

            self.calculating_gradient_steps = batch_size / batch_size_for_calculation


    def get_gradient2(self):

        gradient = 0

        for _ in range(self.calculating_gradient_steps):
            cost_args = self.training_iterator.next()
            gradient += (self.gradient_function(*cost_args)/self.calculating_gradient_steps)

        return gradient


    def SGD_step(self):

        learning_rate = 0.01
        grad = self.get_gradient2()

        current_parameters = self.parameters.get_value()
        direction = - learning_rate * grad
        new_params = current_parameters + direction
        self.parameters.set_value(new_params)


    def train(self):
        '''
        Runs training until a StopTraining exception is raised.

        Training runs indefinitely until one of self.epoch_callbacks raises
        a StopTraining exception.
        '''

        if self._train_called:
            raise RuntimeError("train() has already been called on this %s. "
                               "Re-running train() risks inadvertently "
                               "carrying over implicit state from the "
                               "previous training run, such as the direction "
                               "of parameter updates (via the momentum "
                               "term), or the internal state of the Monitors "
                               "or EpochCallbacks. Instead, instantiate a new "
                               "copy of this %s and run train() on that." %
                               (type(self), type(self)))

        self._train_called = True

        if len(self.epoch_callbacks) == 0:
            raise RuntimeError("self.epoch_callbacks is empty, so Sgd will "
                               "iterate through the training data forever. "
                               "Please add an EpochCallback that will throw a "
                               "StopTraining exception at some point.")

        assert_all_is_instance(self.epoch_callbacks, EpochCallback)

        #
        # End sanity checks
        #

        # Overlaps with self.epoch_callbacks
        iteration_callbacks = [c for c in self.epoch_callbacks
                               if isinstance(c, IterationCallback)]

        try:

            for epoch_callback in self.epoch_callbacks:
                epoch_callback.on_start_training()

            while True:

                for _ in range(self.batches_in_epoch):

                    self.SGD_step()

                print(" ")

                # Epoch callbacks after epoch
                for epoch_callback in self.epoch_callbacks:
                    x = epoch_callback.on_epoch()

                self.epoch_callbacks[-1].callbacks[0](x, None)


        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise
