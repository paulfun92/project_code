"""
Training algorithms, and callbacks for monitoring their progress.
"""

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



class BgfsParameterUpdater(object):

    def __init__(self,
                 parameters,
                 old_parameters,
                 gradient,
                 gradient_at_old_params,
                 learning_rate):


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
                                                concat(parameters.name,
                                                       ' learning rate'))


        self._velocity = make_shared_floatX(
            0.0 * parameters.get_value(),
            concat(parameters.name, ' velocity'),
            broadcastable=parameters.broadcastable)

        '''
        # Try normal SGD first:
        self.momentum = 0.5
        new_velocity = (self.momentum * self._velocity -
                        self.learning_rate * gradient)

        new_parameters = parameters + new_velocity

        self.updates = OrderedDict([(parameters, new_parameters),
                                    (self._velocity, new_velocity)])

        '''
        # k is the current iteration
        # m is how far we want to go back in history for the update
        # s_k = x_{k+1} - x_{k}
        # y_k = gradient_{k+1} - gradient_{k}
        alpha = [None]*k

        q.append(gradient)
        for i in list(reversed(range(k-1,0))):
            alpha[i] = rho[i]*(s[i].T)*q[i+1]
            q[i] = q[i+1] - alpha[i]*y[i]

        r[0] = H0 * q[0]

        n = parameters.get_value().shape[0]


        self.inverse_hessian_estimate = make_shared_floatX(
            numpy.identity(n),
            concat(parameters.name, ' inverse hessian estimate'))

        change_parameter = old_parameters - parameters #d_k
        change_gradient = gradient - gradient_at_old_params

        rho = 1.0/(theano.dot(change_gradient,change_parameter))

        new_inverse_hessian_estimate = ( theano.dot( theano.dot( (numpy.identity(n, dtype=floatX) - rho*theano.tensor.outer(change_parameter,change_gradient)),
                                                    self.inverse_hessian_estimate),(numpy.identity(n, dtype=floatX) - rho*theano.tensor.outer(change_gradient,change_parameter)) )
                                                    + rho*theano.tensor.outer(change_parameter,change_parameter) )

        step_direction = theano.dot(new_inverse_hessian_estimate, gradient)
        new_parameters = parameters - self.learning_rate * step_direction
        new_parameters.name = concat('new ', parameters.name)
        parameters_ = parameters

        self.updates = OrderedDict([(parameters, new_parameters),
                                    (old_parameters, parameters_),
                                    (self.inverse_hessian_estimate, new_inverse_hessian_estimate)])

class Bgfs2(object):


    def __init__(self,
                inputs,
                input_iterator,
                parameters,
                parameter_updater,
                monitors,
                training_set,
                #gradient,
                #scalar_loss_symbol,
                #scalar_loss_symbol_temp,
                epoch_callbacks,
                theano_function_mode=None):


        #
        # Sets members
        #

        self._input_iterator = input_iterator
        self._parameters = parameters
        self._parameter_updater = parameter_updater
        self._monitors = tuple(monitors)

        input_symbols = [i.output_symbol for i in inputs]

        self._compile_update_function_args = \
            {'input_symbols': input_symbols,
             'monitors': self._monitors,
             'parameter_updater': self._parameter_updater,
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

        self.new_epoch = True
        #self.gradient_function = theano.function(input_symbols,gradient)
        #self.objective_value = theano.function(input_symbols,scalar_loss_symbol)
        #self.scalar_loss_symbol_temp = theano.function(input_symbols, scalar_loss_symbol_temp)


    @staticmethod
    def _compile_update_function(input_symbols,
                                 monitors,
                                 parameter_updater,
                                 theano_function_mode):
        '''
        Compiles the function that computes the monitored values.
        '''

        output_symbols = []
        for monitor in monitors:
            output_symbols.extend(monitor.monitored_values)

        updates = OrderedDict()
        updates.update(parameter_updater.updates)

        return theano.function(input_symbols,
                               output_symbols,
                               updates=updates,
                               mode=theano_function_mode)


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

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprop, updates parameters
                # pylint: disable=star-args
                outputs = self._update_function(*cost_arguments)

                #gradient = self.gradient_function(*cost_arguments)
                #obj = self.objective_value(*cost_arguments)
                #obj_temp = self.scalar_loss_symbol_temp(*cost_arguments)

                #print gradient, obj, obj_temp

                # updates monitors
                output_index = 0
                for monitor in self._monitors:
                    new_output_index = (output_index +
                                        len(monitor.monitored_values))
                    assert_less_equal(new_output_index, len(outputs))
                    monitored_values = outputs[output_index:new_output_index]

                    monitor.on_batch(cost_arguments, monitored_values)

                    output_index = new_output_index

                # calls epoch callbacks, if we've iterated through an epoch
                if self._input_iterator.next_is_new_epoch():
                    for callback in all_callbacks:
                        callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise

class Bgfs(object):


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

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprop, updates parameters
                # pylint: disable=star-args
                outputs = self._update_function(*cost_arguments)

                # updates monitors
                output_index = 0
                for monitor in self._monitors:
                    new_output_index = (output_index +
                                        len(monitor.monitored_values))
                    assert_less_equal(new_output_index, len(outputs))
                    monitored_values = outputs[output_index:new_output_index]

                    monitor.on_batch(cost_arguments, monitored_values)

                    output_index = new_output_index

                # calls epoch callbacks, if we've iterated through an epoch
                if self._input_iterator.next_is_new_epoch():
                    for callback in all_callbacks:
                        callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise
