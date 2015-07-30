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



class GdParameterUpdater(ParameterUpdater):

    def __init__(self,
                 parameter,
                 gradient,
                 learning_rate,
                 momentum,
                 use_nesterov):

        #
        # sanity-check args
        #

        assert_is_instance(parameter, theano.tensor.sharedvar.SharedVariable)
        assert_is_instance(gradient, theano.gof.Variable)
        assert_equal(parameter.broadcastable, gradient.broadcastable,
                     "If an Op's .grad() method is buggy, it can return "
                     "broadcast masks.")
        assert_is_subdtype(gradient.dtype, numpy.floating)
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


    	new_velocity = self.momentum* self._velocity - self.learning_rate * self.full_gradient 
    	new_velocity.name = concat('new ', self._velocity.name)


        assert_equal(str(new_velocity.dtype), str(floatX))
        assert_equal(self._velocity.broadcastable, new_velocity.broadcastable)

        step = (self.momentum * new_velocity - self.learning_rate * self.full_gradient
                if use_nesterov
                else new_velocity)

        assert_equal(parameter.broadcastable,
                     step.broadcastable)

        new_parameter = parameter + step
        new_parameter.name = concat('new ', parameter.name)


        updates = OrderedDict([(parameter, new_parameter),
                                (self._velocity, new_velocity),

        total_size_dataset = input_iterator.dataset.tensors[0].shape[0]
        batch_size = input_iterator.batch_size
        steps = total_size_dataset/batch_size

        self.full_gradient_updates = OrderedDict([(self.full_gradient, self.full_gradient + (gradient/steps))])

        super(SemiSgdParameterUpdater, self).__init__(updates)


class Gd(object):


    def __init__(self,
                inputs,
                input_iterator,
                parameters,
                old_parameters,
                parameter_updaters,
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

        total_size_dataset = self._input_iterator.dataset.tensors[0].shape[0]
        batch_size = self._input_iterator.batch_size
        self.batches_in_epoch = total_size_dataset / batch_size

        self._parameters = tuple(parameters)
        self._parameter_updaters = tuple(parameter_updaters)
        self._theano_function_mode = theano_function_mode
        self._inputs = tuple(inputs)

        input_symbols = [i.output_symbol for i in self._inputs]

        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

        self.update_function = self._compile_update_function(input_symbols)
        self.full_gradient_function = self._compile_full_gradient_update_function(input_symbols)


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
                               if isinstance(e, IterationCallback)]

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

        for _ in range(self.batches_in_epoch):

            cost_arguments = self._input_iterator.next()
            self.full_gradient_function(*cost_arguments)


    def Gd_step(self, epoch_counter):

        # Calculate new full gradient:
        self.update_full_gradient()
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
        iteration_callbacks = [c for c in self.epoch_callbacks
                               if isinstance(c, IterationCallback)]

        try:

            for epoch_callback in self.epoch_callbacks:
                epoch_callback.on_start_training()

            while True:

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
                if self._input_iterator.next_is_new_epoch():
                    for epoch_callback in self.epoch_callbacks:
                        epoch_callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise
