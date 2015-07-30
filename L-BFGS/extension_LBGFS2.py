"""
Training algorithms, and callbacks for monitoring their progress.
"""
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
from simplelearn.training import (LimitsNumEpochs, EpochCallback, StopTraining, ParameterUpdater, IterationCallback)
from simplelearn.data.dataset import DataIterator
from simplelearn.nodes import Node
import pdb


class Bgfs(object):


    def __init__(self,
                inputs,
                parameters,
                old_parameters,
                gradient,
                learning_rate,
                training_iterator,
                training_set,
                scalar_loss,
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
        self.old_parameters = old_parameters
        self.training_iterator = training_iterator
        self.learning_rate = learning_rate

        input_symbols = [i.output_symbol for i in inputs]

        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

        self.classification_errors = numpy.asarray([])

        self.gradient_function = theano.function(input_symbols,gradient)

        '''
        output_symbols = []

        iteration_callbacks = [e for e in self.epoch_callbacks
                               if isinstance(e, IterationCallback)]

        for iteration_callback in iteration_callbacks:
            for node_to_compute in iteration_callback.nodes_to_compute:
                output_symbols.append(node_to_compute.output_symbol)

        self.function_outputs = theano.function(input_symbols, output_symbols)

        self.full_training_iterator = training_set.iterator(iterator_type='sequential',
                                                        loop_style='divisible',
                                                        batch_size=50000)
        '''
        self.full_training_iterator = training_set.iterator(iterator_type='sequential',
                                                        loop_style='divisible',
                                                        batch_size=50000)
        self.loss_function = theano.function(input_symbols,scalar_loss)

        self.new_epoch = True
#        self.method = self._parameter_updaters[0].method

        # Initialize saved variables:
        self.rho = []
        self.y = []
        self.s = []
        self.grad_log = []
        self.k = 0 #counter


    def get_full_gradient(self):

        total_size_dataset = self.training_iterator.dataset.tensors[0].shape[0]
        batch_size = self.training_iterator.batch_size
        batches_in_epoch = total_size_dataset / batch_size

        full_gradient = 0

        for _ in range(batches_in_epoch):
            cost_arguments = self.training_iterator.next()
            full_gradient = full_gradient + (self.gradient_function(*cost_arguments) / batches_in_epoch)

        return full_gradient


    def LBFGS_step(self):


        if self.k == 0:
            grad = self.get_full_gradient()
            self.grad_log.append(grad)
        else:
            grad = self.grad_log[-1]

        print(grad)

        current_parameters = self.parameters.get_value()
        n = current_parameters.shape[0]

        alpha = [None]*self.k
        beta = [None]*self.k
        r = [None]*(self.k+1)

        q = grad
        for i in list(reversed(range(0, self.k))):
            alpha[i] = self.rho[i] * numpy.dot(numpy.transpose(self.s[i]),q)
            q = q - alpha[i]*self.y[i]

        if self.k > 0:
            # numpy.identity(n)
            H0 =  ( ( numpy.dot( numpy.transpose(self.y[self.k-1]),self.s[self.k-1] ) )/( numpy.dot( numpy.transpose(self.y[self.k-1]),self.y[self.k-1] ) ) )
            r[0] = H0*q
        else:
            #H0 = numpy.identity(n)
            r[0] = q


        for i in range(self.k):
            beta[i] = self.rho[i] * numpy.dot(numpy.transpose(self.y[i]),r[i])
            r[i+1] = r[i] + numpy.dot( self.s[i], alpha[i] - beta[i] )

        if self.k > 0:
            direction = r[self.k]
        else:
            direction = r[0]

        new_params = current_parameters - self.learning_rate * direction
        print(direction)
        print(direction.shape)
        print(new_params)
        print(new_params.shape)
        new_params = new_params.astype(theano.config.floatX)

        # Update lists:
        self.s.append(new_params - self.parameters.get_value())

        # Take step:
        self.parameters.set_value(new_params)
        self.k = self.k + 1 # Update counter

        new_grad = self.get_full_gradient()
        self.grad_log.append(new_grad)
        self.y.append(new_grad - grad)
        temp = numpy.dot( numpy.transpose(self.y[-1]),self.s[-1] )
        new_rho = 1.0/temp
        self.rho.append(new_rho)

        '''
        #Try normal SGD:
        grad = self.get_gradient(cost_arguments)
        parameter_value = self.parameters.get_value()
        new_param = parameter_value - self.learning_rate * grad
        self.parameters.set_value(new_param)
        '''


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

                self.LBFGS_step()

                '''
                cost_arguments = self.full_training_iterator.next()
                all_callback_outputs = self.function_outputs(*cost_arguments)

                output_index = 0
                for iteration_callback in iteration_callbacks:
                    num_outputs = len(iteration_callback.nodes_to_compute)
                    new_output_index = output_index + num_outputs

                    assert_less_equal(new_output_index,
                                      len(all_callback_outputs))

                    outputs = \
                        all_callback_outputs[output_index:new_output_index]

                    iteration_callback.on_iteration(outputs)

                    output_index = new_output_index

                assert_equal(output_index, len(all_callback_outputs))
                '''
                print(" ")
                cost_args = self.full_training_iterator.next()
                print(self.loss_function(*cost_args))
                # if we've iterated through an epoch, call epoch_callbacks'
                # on_epoch() methods.
                #if self.training_iterator.next_is_new_epoch():
                for epoch_callback in self.epoch_callbacks:
                    epoch_callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise
