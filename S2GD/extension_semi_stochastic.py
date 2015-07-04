"""
Training algorithms, and callbacks for monitoring their progress.
"""


from __future__ import print_function

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"

import os
import copy
import warnings
import cPickle
from collections import Sequence, OrderedDict
import numpy
import theano
import theano.tensor as T
from nose.tools import (assert_true,
                        assert_equal,
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
                                 assert_is_subdtype)
from simplelearn.training import (SavesAtMinimum, AverageMonitor, ValidationCallback, SavesAtMinimum, StopsOnStagnation,
                                  LimitsNumEpochs, EpochCallback, Monitor, StopTraining)
from simplelearn.data.dataset import DataIterator
from simplelearn.utils import safe_izip
from simplelearn.formats import Format
from simplelearn.nodes import Node
import pdb



class SemiSgdParameterUpdater(object):

    def __init__(self,
                 parameter,
                 gradient,
                 gradient_at_old_params,
                 learning_rate,
                 momentum,
                 method,
                 input_iterator,
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
            self.updates = OrderedDict([(parameter, new_parameter),
                                        (self._velocity, new_velocity),
                                        (self.full_gradient, updated_full_gradient)])
        else:
            self.updates = OrderedDict([(parameter, new_parameter),
                                        (self._velocity, new_velocity)])

class SemiSgd(object):


    def __init__(self,
                inputs,
                input_iterator,
                parameters,
                old_parameters,
                parameter_updaters,
                gradient,
                monitors,
                training_set,
                epoch_callbacks,
                theano_function_mode=None):


        #
        # sanity-checks the arguments.
        #

        assert_is_instance(inputs, Sequence)
        for input in inputs:
            assert_is_instance(input, Node)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        for (input,
             iterator_input) in safe_izip(inputs,
                                          input_iterator.make_input_nodes()):
            assert_equal(input.output_format, iterator_input.output_format)

        assert_is_instance(parameters, Sequence)
        for parameter in parameters:
            assert_is_instance(parameter,
                               theano.tensor.sharedvar.SharedVariable)

        assert_is_instance(monitors, Sequence)
        for monitor in monitors:
            assert_is_instance(monitor, Monitor)

        assert_is_instance(epoch_callbacks, Sequence)
        for epoch_callback in epoch_callbacks:
            assert_is_instance(epoch_callback, EpochCallback)
            if isinstance(epoch_callback, Monitor):
                warnings.warn("You've passed a Monitor subclass %s "
                              "as one of the epoch_callbacks. If you want the "
                              ".on_batch() method to be called on this, you "
                              "need to pass it in as one of the monitors." %
                              str(epoch_callback))

        #
        # Sets members
        #

        self._input_iterator = input_iterator
        self._parameters = tuple(parameters)
        self._old_parameters = tuple(old_parameters)
        self._parameter_updaters = tuple(parameter_updaters)
        self._monitors = tuple(monitors)

        input_symbols = [i.output_symbol for i in inputs]

        self._compile_update_function_args = \
            {'input_symbols': input_symbols,
             'monitors': self._monitors,
             'parameter_updaters': self._parameter_updaters,
             'theano_function_mode': theano_function_mode}

        self._update_function = self._compile_update_function(
            **self._compile_update_function_args)

        repeated_callbacks = frozenset(monitors).intersection(epoch_callbacks)
        assert_equal(len(repeated_callbacks),
                     0,
                     "There were duplicate entries between monitors and "
                     "epoch_callbacks: %s" % str(repeated_callbacks))

        # These get called once before any training, and after each epoch
        # thereafter. One of them must halt the training at some point by
        # throwing a StopTraining exception.
        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

        self._training_full_gradient_iterator = training_set.iterator(iterator_type='sequential', batch_size=50000) #training_set.size
        self.classification_errors = numpy.asarray([])

        self.gradient_function = theano.function(input_symbols,gradient)
        self.new_epoch = True
        self.method = self._parameter_updaters[0].method


    @staticmethod
    def _compile_update_function(input_symbols,
                                 monitors,
                                 parameter_updaters,
                                 theano_function_mode):
        '''
        Compiles the function that computes the monitored values.
        '''

        output_symbols = []
        for monitor in monitors:
            output_symbols.extend(monitor.monitored_values)

        updates = OrderedDict()
        for updater in parameter_updaters:
            assert_is_instance(updater.updates, OrderedDict)
            updates.update(updater.updates)

        return theano.function(input_symbols,
                               output_symbols,
                               updates=updates,
                               mode=theano_function_mode)


    def get_gradient(self, cost_arguments):

        return self.gradient_function(*cost_arguments)
        # Function that returns gradient of the loss function w.r.t the parameters W and b,
        # at point of current parameters, given the input batch of images.


    def semi_sgd_step(self, epoch_counter):

        # If new epoch is started:
        if self.new_epoch == True and self.method is not 'SGD':

            # Set the old parameters (x_j) equal to the current parameters (y_(j,t)):
            for i in range(len(self._old_parameters)):
                self._old_parameters[i].set_value(self._parameters[i].get_value())

            if self.method == 'S2GD' or (self.method == 'S2GD_plus' and epoch_counter > 1) or (self.method == 'S2GD_rolling' and epoch_counter == 1): # and epoch_counter == 1):
                # Calculate the new full gradient:
                cost_arguments = self._training_full_gradient_iterator.next()
                new_full_gradient = self.get_gradient(cost_arguments)
                for i in range(len(self._parameter_updaters)):
                    self._parameter_updaters[i].full_gradient.set_value(new_full_gradient[i])

            self.new_epoch = False

        # Take a step of the inner loop:
        # gets batch of data
        cost_arguments = self._input_iterator.next()

        # Take the step here:
        outputs = self._update_function(*cost_arguments)

        return outputs, cost_arguments

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

        if len(self.epoch_callbacks) + len(self._monitors) == 0:
            raise RuntimeError("self._monitors and self.epoch_callbacks are "
                               "both empty, so this will "
                               "iterate through the training data forever. "
                               "Please add an EpochCallback or "
                               "Monitor that will throw a "
                               "StopTraining exception at some point.")

        #
        # End sanity checks
        #

        try:
            all_callbacks = self._monitors + tuple(self.epoch_callbacks)
            for callback in all_callbacks:
                callback.on_start_training()
                #if classification_error is not None:
                    #self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

            # Set initial parameters for SemiSGD:
            # max_stochastic_steps_per_epoch = max # of stochastic steps per epoch
            # v = lower bound on the constant of the strongly convex loss function
            # stochastic_steps = # of stochastic steps taken in an epoch, calculated geometrically.

            total_size_dataset = self._input_iterator.dataset.tensors[0].shape[0]
            batch_size = self._input_iterator.batch_size
            stochastic_steps = total_size_dataset/batch_size
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
                            stochastic_steps = t
                            break

                # Run the semi-stochastic gradient descent main loop
                for t in range(stochastic_steps):
                    # Now take a step:
                    outputs, cost_arguments = self.semi_sgd_step(epoch_counter)

                    # updates monitors
                    output_index = 0
                    for monitor in self._monitors:
                        new_output_index = (output_index +
                                            len(monitor.monitored_values))
                        assert_less_equal(new_output_index, len(outputs))
                        monitored_values = outputs[output_index:new_output_index]

                        monitor.on_batch(cost_arguments, monitored_values)

                        output_index = new_output_index

                    self._input_iterator.next_is_new_epoch()

                self.new_epoch = True
                epoch_counter = epoch_counter + 1
                if epoch_counter == 2 and self.method == 'S2GD_plus':
                    for updater in self._parameter_updaters:
                        updater.S2GD_on.set_value(1.0)

                # calls epoch callbacks, if we've iterated through an epoch
                # if self._input_iterator.next_is_new_epoch():
                for callback in all_callbacks:
                    callback.on_epoch()
                    #if classification_error is not None:
                        #self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise
