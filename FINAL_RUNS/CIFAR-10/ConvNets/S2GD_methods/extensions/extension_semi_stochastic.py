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
from simplelearn.training import (LimitsNumEpochs, EpochCallback, StopTraining, ParameterUpdater, IterationCallback, EpochTimer2)
from simplelearn.data.dataset import DataIterator
from simplelearn.nodes import Node
import pdb



class SemiSgdParameterUpdater(ParameterUpdater):

    def __init__(self,
                 parameter,
                 gradient,
                 gradient_at_old_params,
                 learning_rate,
                 momentum,
                 method,
                 input_iterator,
                 input_iterator_full,
                 use_nesterov):

        #
        # sanity-check args
        #

        assert_is_instance(parameter, theano.tensor.sharedvar.SharedVariable)
        assert_is_instance(gradient, theano.gof.Variable)
        assert_is_instance(gradient_at_old_params, theano.gof.Variable)
        assert_equal(parameter.broadcastable, gradient.broadcastable,
                     "If an Op's .grad() method is buggy, it can return "
                     "broadcast masks.")
        assert_is_subdtype(gradient.dtype, numpy.floating)
        assert_is_subdtype(gradient_at_old_params.dtype, numpy.floating)
        assert_greater_equal(learning_rate, 0)
        assert_greater_equal(momentum, 0)
        assert_is_instance(use_nesterov, bool)

        floatX = theano.config.floatX

        if str(gradient.dtype) != str(floatX):
            gradient = theano.tensor.cast(gradient, floatX)

        #
        # define updates, set members
        #

        def concat(str0, str1):
            '''
            Like str0 + str1, except returns None if either is None.
            '''
            if str0 is None or str1 is None:
                return None
            else:
                return str0 + str1

        def make_shared_floatX(numeric_var, name, **kwargs):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),
                                 name=name,
                                 **kwargs)

        self.learning_rate = make_shared_floatX(learning_rate,
                                                concat(parameter.name,
                                                       ' learning rate'))

        self.momentum = make_shared_floatX(momentum,
                                           concat(parameter.name, ' momentum'))

        self._velocity = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' velocity'),
            broadcastable=parameter.broadcastable)

        self.full_gradient = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' full gradient'),
            broadcastable=parameter.broadcastable)

        # This variable takes value 1 if S2GD is used and 0 otherwise.
        self.method = method
        if self.method == 'SGD' or self.method == 'S2GD_plus':
            multiplier = 0.0
        elif self.method == 'S2GD' or self.method == 'S2GD_rolling':
            multiplier = 1.0
        else:
            raise Exception('Please enter a valid method: "SGD", "S2GD", "S2GD_plus", or "S2GD_rolling"')

        self.S2GD_on = make_shared_floatX(numeric_var=multiplier, name='use_S2GD')

        # updated_full_gradient = 0
        if self.method == 'S2GD_rolling':
            total_size_dataset = float(input_iterator.dataset.tensors[0].shape[0])
            batch_size = float(input_iterator.batch_size)
            updated_full_gradient = (gradient*batch_size + self.full_gradient*total_size_dataset - gradient_at_old_params*batch_size)/ total_size_dataset
            new_velocity = self.momentum* self._velocity - self.learning_rate * updated_full_gradient
            new_velocity.name = concat('new ', self._velocity.name)
        else:
            new_velocity = self.momentum* self._velocity - self.learning_rate * (gradient + self.S2GD_on * (self.full_gradient - gradient_at_old_params))
            new_velocity.name = concat('new ', self._velocity.name)


        assert_equal(str(new_velocity.dtype), str(floatX))
        assert_equal(self._velocity.broadcastable, new_velocity.broadcastable)

        step = (self.momentum * new_velocity - self.learning_rate * gradient
                if use_nesterov
                else new_velocity)

        assert_equal(parameter.broadcastable,
                     step.broadcastable)

        new_parameter = parameter + step
        new_parameter.name = concat('new ', parameter.name)

        # self.updates = 0
        if self.method == 'S2GD_rolling':
            updates = OrderedDict([(parameter, new_parameter),
                                        (self._velocity, new_velocity),
                                        (self.full_gradient, updated_full_gradient)])
        else:
            updates = OrderedDict([(parameter, new_parameter),
                                        (self._velocity, new_velocity)])

        total_size_dataset = input_iterator_full.dataset.tensors[0].shape[0]
        batch_size = input_iterator_full.batch_size
        steps = total_size_dataset/batch_size

        self.full_gradient_updates = OrderedDict([(self.full_gradient, self.full_gradient + (gradient/steps))])

        super(SemiSgdParameterUpdater, self).__init__(updates)

class SemiSgd(object):


    def __init__(self,
                inputs,
                input_iterator,
                parameters,
                old_parameters,
                parameter_updaters,
                iterator_full_gradient,
                epoch_callbacks,
                theano_function_mode=None):


        #
        # sanity-checks the arguments.
        #

        assert_all_is_instance(inputs, Node)
        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        for (input,
             iterator_input) in safe_izip(inputs,
                                          input_iterator.make_input_nodes()):
            assert_equal(input.output_format, iterator_input.output_format)

        assert_equal(len(epoch_callbacks),
                     len(frozenset(epoch_callbacks)),
                     "There were duplicate callbacks.")

        assert_all_is_instance(epoch_callbacks, EpochCallback)


        #
        # Sets members
        #

        self._input_iterator = input_iterator
        self._parameters = tuple(parameters)
        self._old_parameters = tuple(old_parameters)
        self._parameter_updaters = tuple(parameter_updaters)
        self._theano_function_mode = theano_function_mode
        self._inputs = tuple(inputs)

        input_symbols = [i.output_symbol for i in self._inputs]

        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

        self.new_epoch = True
        self.method = self._parameter_updaters[0].method
        self.update_function = self._compile_update_function(input_symbols)
        self.full_gradient_function = self._compile_full_gradient_update_function(input_symbols)

        self.full_gradient_iterator = iterator_full_gradient
        total_size_dataset = self.full_gradient_iterator.dataset.tensors[0].shape[0]
        batch_size = self.full_gradient_iterator.batch_size
        self.batches_in_epoch_full = total_size_dataset/batch_size


    def _compile_full_gradient_update_function(self,input_symbols):

        output_symbols = []
        update_pairs = OrderedDict()

        for param_updater in self._parameter_updaters:
            update_pairs.update(param_updater.full_gradient_updates)

        return theano.function(input_symbols,
                               output_symbols,
                               updates=update_pairs,
                               mode=self._theano_function_mode)


#    @staticmethod
    def _compile_update_function(self,input_symbols):

        iteration_callbacks = [e for e in self.epoch_callbacks
                               if (isinstance(e, IterationCallback) and not isinstance(e, EpochTimer2))]

        output_symbols = []

        for iteration_callback in iteration_callbacks:
            for node_to_compute in iteration_callback.nodes_to_compute:
                output_symbols.append(node_to_compute.output_symbol)

        update_pairs = OrderedDict()

        for iteration_callback in iteration_callbacks:
            update_pairs.update(iteration_callback.update_pairs)

        return theano.function(input_symbols,
                               output_symbols,
                               updates=update_pairs,
                               mode=self._theano_function_mode)


    def update_full_gradient(self):

        for i in range(len(self._parameter_updaters)):
            self._parameter_updaters[i].full_gradient.set_value(0*self._parameter_updaters[i].full_gradient.get_value())

        for _ in range(self.batches_in_epoch_full):

            cost_arguments = self.full_gradient_iterator.next()
            self.full_gradient_function(*cost_arguments)


    def semi_sgd_step(self, epoch_counter):

        # If new epoch is started:
        if self.new_epoch == True and self.method is not 'SGD':

            # Set the old parameters (x_j) equal to the current parameters (y_(j,t)):
            for i in range(len(self._old_parameters)):
                self._old_parameters[i].set_value(self._parameters[i].get_value())

            if self.method == 'S2GD' or (self.method == 'S2GD_plus' and epoch_counter > 1) or (self.method == 'S2GD_rolling' and epoch_counter == 1): # and epoch_counter == 1):
                # Update full gradient:
                self.update_full_gradient()

            self.new_epoch = False

        # Take a step of the inner loop:
        # gets batch of data
        cost_arguments = self._input_iterator.next()

        # Take the step here:
        all_callback_outputs = self.update_function(*cost_arguments)

        return all_callback_outputs

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
        iteration_callbacks = [e for e in self.epoch_callbacks
                       if (isinstance(e, IterationCallback) and not isinstance(e, EpochTimer2))]

        try:

            for epoch_callback in self.epoch_callbacks:
                epoch_callback.on_start_training()

            # Set initial parameters for SemiSGD:
            # max_stochastic_steps_per_epoch = max # of stochastic steps per epoch
            # v = lower bound on the constant of the strongly convex loss function
            # stochastic_steps = # of stochastic steps taken in an epoch, calculated geometrically.

            total_size_dataset = self._input_iterator.dataset.tensors[0].shape[0]
            batch_size = self._input_iterator.batch_size
            self.stochastic_steps = total_size_dataset/batch_size
            epoch_counter = 1

            if self.method == 'S2GD':
                max_stochastic_steps_per_epoch = total_size_dataset/batch_size
                v = 0.05
                learning_rate = self._parameter_updaters[0].learning_rate.get_value()
                # Calculate the sum of the probabilities for geometric distribution:
                sum = 0
                for t in range(1,max_stochastic_steps_per_epoch+1):
                    add = pow((1-v*learning_rate),(max_stochastic_steps_per_epoch - t))
                    sum = sum + add

            while True:

                if self.method == 'S2GD':
                    # Determine # of stochastic steps taken in the epoch:

                    cummulative_prob = 0
                    rand = numpy.random.uniform(0,1)
                    for t in range(1,max_stochastic_steps_per_epoch+1):
                        prob = pow((1-v*learning_rate),(max_stochastic_steps_per_epoch - t)) / sum
                        cummulative_prob = cummulative_prob + prob
                        if  rand < cummulative_prob:
                            self.stochastic_steps = t
                            break

                # Run the semi-stochastic gradient descent main loop
                for t in range(self.stochastic_steps):
                    # Now take a step:
                    all_callback_outputs = self.semi_sgd_step(epoch_counter)

                    # calls iteration_callbacks' on_iteration() method, passing
                    # in their output values, if any.
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

                    # if we've iterated through an epoch, call epoch_callbacks'
                    # on_epoch() methods.
                    #if self._input_iterator.next_is_new_epoch():
                for epoch_callback in self.epoch_callbacks:
                    x = epoch_callback.on_epoch()

                self.epoch_callbacks[-1].callbacks[0](x, None)
                self.new_epoch = True
                epoch_counter += 1

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise
