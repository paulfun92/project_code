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


class RMSpropSgdParameterUpdater(object):
    '''
    Defines how to update parameters using SGD with momentum.

    You can set the learning rate and momentum dynamically during the
    optimization.

    Fields
    ------
    learning_rate: theano.tensor.SharedScalarVariable
      Call set_value() on this to change the learning rate.

    momentum:  theano.tensor.SharedScalarVariable
      Call set_value() on this to change the momentum.

    updates: dict
      A dictionary with (var: new_var) pairs, where var and new_var are
      Theano expressions. At each training update, var's value will be
      replaced with new_var.

      This contains the update for not just a parameter, but also the internal
      state, such as the as the momentum-averaged update direction.
    '''

    def __init__(self,
                 parameter,
                 gradient,  # see (*) below
                 learning_rate,
                 momentum,
                 use_nesterov):

        # (*): We pass in the gradient, rather than the cost, since there are
        # different ways to generate the gradient expression, and we want to
        # allow the user to choose different ones, rather than generating the
        # gradient here ourselves. In particular, the 'consider_constant'
        # argument to theano.gradient.grad() could be of interest to the user.
        # (It's a list of symbols to consider constant, and thus not
        # backpropagate through.)
        '''
        Parameters
        ----------
        parameter: A theano symbol
          A parameter being optimized by an Sgd trainer.

        gradient: A theano symbol
          The gradient of the loss function w.r.t. the above parameter.

        learing_rate: float
          The initial value of the learning rate.

        momentum: float
          A parameter affecting how smeared the update direction is over
          multiple batches. Use 0.0 for momentum-less SGD.

        use_nesterov: bool
          If true, use Nesterov momentum. (See "Advances in Optimizing
          Recurrent Networks", Yoshua Bengio, et al.)
        '''

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

        decay_rate = 0.1
        self.decay_rate = make_shared_floatX(decay_rate,
                                           concat(parameter.name, ' decay rate'))

        self._velocity = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' velocity'),
            broadcastable=parameter.broadcastable)

        self.mean_square = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, '  MeanSquare'),
            broadcastable=parameter.broadcastable)


        new_mean_square = self.decay_rate * self.mean_square + (1-self.decay_rate) * pow(gradient,2)
        new_mean_square.name = concat('new ', self.mean_square.name)

        new_velocity = (self.momentum * self._velocity -
                        self.learning_rate * (gradient / (pow(new_mean_square, 0.5) + 0.6)) )
        new_velocity.name = concat('new ', self._velocity.name)

        assert_equal(str(new_velocity.dtype), str(floatX))
        assert_equal(self._velocity.broadcastable, new_velocity.broadcastable)

        step = (self.momentum * new_velocity - self.learning_rate * gradient
                if use_nesterov
                else new_velocity)

        #step2 = self.learning_rate * (gradient / ( pow(new_mean_square, 0.5) + 0.6) )

        assert_equal(parameter.broadcastable,
                     step.broadcastable)

        new_parameter = parameter + step
        new_parameter.name = concat('new ', parameter.name)

        self.updates = OrderedDict([(parameter, new_parameter),
                                    (self._velocity, new_velocity),
                                    (self.mean_square, new_mean_square)])


class RMSpropSgd(object):

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
                 input_iterator,
                 parameters,
                 parameter_updaters,
                 monitors,
                 epoch_callbacks,
                 theano_function_mode=None):

        '''
        Parameters
        ----------

        inputs: sequence of Nodes.
          Symbols for the outputs of the input_iterator.
          These should come from input_iterator.make_input_nodes()

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of training set batches, such as (values, labels).

        parameters: sequence of theano.tensor.sharedvar.SharedVariables
          What this trainer modifies to lower the cost. These are typically
          model weights, though they could also be inputs (e.g. for optimizing
          input images).

        parameter_updaters: sequence of SgdParameterUpdaters
          updaters for the corresponding elements in <parameters>.
          These are defined using the loss function to be minimized.

        monitors: (optional) sequence of Monitors.
          These are also used as epoch callbacks.

        epoch_callbacks: sequence of EpochCallbacks
          One of these must throw a StopTraining exception for the training to
          halt.

        theano_function_mode: theano.compile.Mode
          Optional. The 'mode' argument to pass to theano.function().
          An example: pylearn2.devtools.nan_guard.NanGuard()
        '''

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
        assert_is_instance(parameter_updaters, Sequence)
        for parameter, updater in safe_izip(parameters, parameter_updaters):
            assert_is_instance(parameter,
                               theano.tensor.sharedvar.SharedVariable)

            assert_is_instance(updater, RMSpropSgdParameterUpdater)

            assert_in(parameter, updater.updates)

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
                return
            else:
                raise