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
from simplelearn.training import (LimitsNumEpochs, EpochCallback, StopTraining, ParameterUpdater, IterationCallback, ValidationCallback, MeanOverEpoch)
from simplelearn.data.dataset import DataIterator
from simplelearn.nodes import Node
import pdb


class CG(object):


    def __init__(self,
                inputs,
                parameters,
                gradient,
                learning_rate,
                training_iterator,
                validation_iterator,
                scalar_loss,
                armijo,
                tangent,
                method,
                batch_size,
                epoch_callbacks,
                param_shapes=None):


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

        self.armijo = armijo
        self.tangent = tangent
        self.parameters = parameters
        self.training_iterator = training_iterator
        self.validation_iterator = validation_iterator
        self.learning_rate = learning_rate
        self.method = method

        input_symbols = [i.output_symbol for i in inputs]
        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

        self.classification_errors = numpy.asarray([])
        self.gradient_function = theano.function(input_symbols,gradient)
        self.loss_function = theano.function(input_symbols,scalar_loss)

        self.new_epoch = True
#        self.method = self._parameter_updaters[0].method

        self.param_shapes = param_shapes
        # Initialize saved variables:
        self.k = 0 #counter

        '''
        total_size_dataset_full = self.full_training_iterator.dataset.tensors[0].shape[0]
        batch_size_full =self.full_training_iterator.batch_size
        self.batches_in_epoch_full_gradient = total_size_dataset_full / batch_size_full
        '''

        total_size_dataset = self.training_iterator.dataset.tensors[0].shape[0]
        self.batches_in_epoch = total_size_dataset / batch_size

        batch_size_for_calculation =self.training_iterator.batch_size
        assert_less_equal(batch_size_for_calculation, batch_size)

        self.calculating_gradient_steps = batch_size / batch_size_for_calculation

        total_size_validation_dataset = self.validation_iterator.dataset.tensors[0].shape[0]
        batch_size_validation = self.validation_iterator.batch_size
        self.batches_in_epoch_validation = total_size_validation_dataset / batch_size_validation

        if self.armijo == True:
            self.function_value = theano.function(input_symbols, scalar_loss)
            self.validation_function_value_log = []
            validation_function_value = self.get_validation_function_value()
            self.validation_function_value_log.append(validation_function_value)

        self.previous_gradient = 0
        self.previous_direction = 0


    def get_gradient2(self):

        gradient = 0

        for _ in range(self.calculating_gradient_steps):
            cost_args = self.training_iterator.next()
            gradient += (self.gradient_function(*cost_args)/self.calculating_gradient_steps)

        return gradient

    def get_validation_function_value(self):

        validation_function_value = 0
        # Get validation cost function value
        for _ in range(self.batches_in_epoch_validation):
            cost_args = self.validation_iterator.next()
            validation_function_value += (self.function_value(*cost_args) / self.batches_in_epoch_validation)

        print validation_function_value
        return validation_function_value


    def CG_step(self):

        # Initialize parameters for backtracking:
        tau = 0.5
        beta_ = 1E-4
        tolerance = 1E-6
        learning_rate = self.learning_rate
        alpha_found = False

        if self.armijo == True:
            validation_function_value_previous = self.validation_function_value_log[-1]

        grad = self.get_gradient2()

        if self.k == 0:
            print("Conjugate Gradient is now carried out.")
            direction = -grad
            self.previous_gradient = grad
            self.previous_direction = direction
        else:
            if self.method == 'FR':
                beta = numpy.dot(grad,grad) / numpy.dot(self.previous_gradient, self.previous_gradient)
            elif self.method == 'PR':
                beta = numpy.dot(grad, (grad-self.previous_gradient)) / numpy.dot(self.previous_gradient, self.previous_gradient)
            elif self.method == 'HS':
                beta = numpy.dot(grad, (grad-self.previous_gradient)) / numpy.dot(self.previous_direction, (grad-self.previous_gradient))
            elif self.method == 'DY':
                beta = numpy.dot(grad,grad) / numpy.dot(self.previous_direction, (grad-self.previous_gradient))
            else:
                raise Exception('Please provide a valid variant of Conjugate Gradient.')

            direction = -grad + beta*self.previous_direction

            #update:
            self.previous_gradient = grad
            self.previous_direction = direction

        current_parameters = self.parameters.get_value()

        if self.armijo == True: # Perform line search

            inner_loop_counter = 0
            while not alpha_found:

                new_params = current_parameters + learning_rate * direction
                new_params = new_params.astype(theano.config.floatX)
                #direction_ = new_params - current_parameters

                # Take step:
                self.parameters.set_value(new_params)
                validation_function_value = self.get_validation_function_value()

                val2 = validation_function_value_previous - (beta_ * learning_rate * numpy.dot(grad,direction))
                # Do the actual backtracking:
                if validation_function_value < val2+tolerance or inner_loop_counter > 15:
                    alpha_found = True
                else:
                    learning_rate = tau*learning_rate
                    inner_loop_counter += 1

            # Learning rate FOUND

        else:
            new_params = current_parameters + learning_rate * direction
            if not self.tangent:
                new_params = new_params.astype(theano.config.floatX)
                self.parameters.set_value(new_params)

        if self.tangent == True:
            # Use tangent approach:
            index_from = 0
            for shape in self.param_shapes:
                if len(shape) == 4: # it is a filter
                    col_length = shape[1]*shape[2]*shape[3]
                    for _ in range(shape[0]):
                        index_to = index_from + col_length
                        w = current_parameters[index_from:index_to]
                        d = new_params[index_from:index_to] - w
                        update = d - (numpy.dot(d,w)/numpy.power(numpy.linalg.norm(w),2))*w
                        new_params[index_from:index_to] = w + update
                        index_from = index_to
                elif len(shape) == 2:
                    col_length = shape[0]
                    for _ in range(shape[1]):
                        index_to = index_from + col_length
                        w = current_parameters[index_from:index_to]
                        d = new_params[index_from:index_to] - w
                        update = d - (numpy.dot(d,w)/numpy.power(numpy.linalg.norm(w),2))*w
                        new_params[index_from:index_to] = w + update
                        index_from = index_to
                else:
                    index_from += numpy.product(shape)

            new_params = new_params.astype(theano.config.floatX)
            self.parameters.set_value(new_params)

        if self.armijo == True:
            # Save validation function value for next time to save time
            if self.tangent:
                validation_function_value = self.get_validation_function_value()
            self.validation_function_value_log.append(validation_function_value)

            # Learning rate is found
            print("learning rate used: ", learning_rate)


        self.k = self.k + 1 # Update counter


    def SGD_step_armijo(self):

        tau = 0.5
        beta_ = 1E-4
        tolerance = 1E-6
        alpha_found = False
        validation_function_value_previous = self.validation_function_value

        learning_rate = 0.5
        grad = self.get_gradient2()

        current_parameters = self.parameters.get_value()

        while not alpha_found:

            direction = - learning_rate * grad
            new_params = current_parameters + direction
            new_params = new_params.astype(theano.config.floatX)
            self.parameters.set_value(new_params)

            self.validation_function_value = self.get_validation_function_value()

            val2 = validation_function_value_previous - (beta_ * learning_rate * numpy.dot(grad,direction))
            # Do the actual backtracking:
            if self.validation_function_value < val2+tolerance:
                alpha_found = True
                print "Alpha used:", learning_rate
            else:
                learning_rate = tau*learning_rate


    def SGD_step(self):

        learning_rate = 0.5
        grad = self.get_gradient2()

        current_parameters = self.parameters.get_value()
        direction = - learning_rate * grad
        new_params = current_parameters + direction
        self.parameters.set_value(new_params)

        '''
        self.s.append(direction)
        self.k = self.k + 1 # Update counter
        temp = numpy.dot( numpy.transpose(self.y[-1]),self.s[-1] )
        new_rho = 1.0/temp
        self.rho.append(new_rho)

        validation_function_value = self.get_validation_function_value()
        self.validation_function_value_log.append(validation_function_value)
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

                for _ in range(self.batches_in_epoch):
                    self.CG_step()

                '''
                self.validation_function_value = self.get_validation_function_value()
                #self.validation_function_value_log.append(validation_function_value)

                for epoch_callback in self.epoch_callbacks:
                    epoch_callback.on_epoch()

                for _ in range(batches_in_epoch):
                    self.LBFGS_step()
                '''

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

                '''
                cost_args = self.full_training_iterator.next()
                print(self.loss_function(*cost_args))
                '''

                # if we've iterated through an epoch, call epoch_callbacks'
                # on_epoch() methods.
                #if self.training_iterator.next_is_new_epoch():
                for epoch_callback in self.epoch_callbacks:
                    x = epoch_callback.on_epoch()

                self.epoch_callbacks[-1].callbacks[0](x, None)

                '''
                for _ in range(batches_in_epoch):
                    self.SGD_step_armijo()

                for epoch_callback in self.epoch_callbacks:
                    epoch_callback.on_epoch()
                '''


        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise
