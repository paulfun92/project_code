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


class SgdParameterUpdater(ParameterUpdater):
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
                 learning_rate):

        # (*): We pass in the gradient, rather than the cost, since there are
        # different ways to generate the gradient expression, and we want to
        # allow the user to choose different ones, rather than generating the
        # gradient here ourselves. In particular, the 'consider_constant'
        # argument to theano.gradient.grad() could be of interest to the user.
        # (It's a list of symbols to consider constant, and thus not
        # backpropagate through to their inputs.)
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

        self.param_length = parameter.get_value().shape[0]

        self.learning_rate = make_shared_floatX(learning_rate,
                                                concat(parameter.name,
                                                       ' learning rate'))
        self.previous_direction = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' velocity'),
            broadcastable=parameter.broadcastable)

        self.previous_gradient = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' velocity'),
            broadcastable=parameter.broadcastable)


        beta = theano.dot(gradient,gradient) / theano.dot(self.previous_gradient, self.previous_gradient)

        new_direction = -1*gradient + beta * self.previous_direction

        new_parameter = parameter + self.learning_rate * new_direction
        #new_parameter.name = concat('new ', parameter.name)

        #new_parameter = parameter - self.learning_rate * gradient

        updates = OrderedDict([(self.previous_direction, new_direction),
                                (self.previous_gradient, gradient),
                                (parameter, new_parameter)])

        new_direction_ = -1*gradient
        new_parameter_ = parameter + self.learning_rate * new_direction_

        self.updates2 = OrderedDict([(self.previous_direction, new_direction_),
                                (self.previous_gradient, gradient),
                                (parameter, new_parameter_)])

        super(SgdParameterUpdater, self).__init__(updates)


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
                 input_iterator,
                 parameter_updater,
                 callbacks,
                 theano_function_mode=None):

        '''
        Parameters
        ----------

        inputs: sequence of Nodes.
          Symbols for the outputs of the input_iterator.
          These should come from input_iterator.make_input_nodes()

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of training set batches, such as (values, labels).

        callbacks: Sequence of EpochCallbacks
          This includes subclasses like IterationCallback &
          ParameterUpdater. One of these callbacks must throw a StopTraining
          exception for the training to halt.

        theano_function_mode: theano.compile.Mode
          Optional. The 'mode' argument to pass to theano.function().
          An example: pylearn2.devtools.nan_guard.NanGuard()
        '''

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

        assert_equal(len(callbacks),
                     len(frozenset(callbacks)),
                     "There were duplicate callbacks.")

        assert_all_is_instance(callbacks, EpochCallback)

        #
        # Sets members
        #

        self._inputs = inputs
        self.parameter_updater = parameter_updater
        self._input_iterator = input_iterator
        self._theano_function_mode = theano_function_mode
        self.epoch_callbacks = list(callbacks)
        self._train_called = False

    def _compile_update_function(self):
        input_symbols = [i.output_symbol for i in self._inputs]

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


    def _compile_update_function2(self):
        input_symbols = [i.output_symbol for i in self._inputs]

        iteration_callbacks = [e for e in self.epoch_callbacks
                               if (isinstance(e, IterationCallback) and not isinstance(e, EpochTimer2))]

        output_symbols = []
        for iteration_callback in iteration_callbacks:
            for node_to_compute in iteration_callback.nodes_to_compute:
                output_symbols.append(node_to_compute.output_symbol)

        update_pairs2 = self.parameter_updater.updates2

        return theano.function(input_symbols,
                               output_symbols,
                               updates=update_pairs2,
                               mode=self._theano_function_mode)


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

        update_function = self._compile_update_function()
        update_function2 = self._compile_update_function2()

        # Overlaps with self.epoch_callbacks
        iteration_callbacks = [e for e in self.epoch_callbacks
                               if (isinstance(e, IterationCallback) and not isinstance(e, EpochTimer2))]

        try:
            for epoch_callback in self.epoch_callbacks:
                epoch_callback.on_start_training()

            in_batch_counter = 0
            iteration_counter = 0
            while True:

                if in_batch_counter == 0:
                    cost_arguments = self._input_iterator.next()
                    all_callback_outputs = update_function2(*cost_arguments)
                else:
                    all_callback_outputs = update_function(*cost_arguments)

                if in_batch_counter > 2:
                    # get new batch of data
                    in_batch_counter = -1
                    self.parameter_updater.previous_gradient.set_value(numpy.zeros(self.parameter_updater.param_length, dtype = theano.config.floatX))
                    self.parameter_updater.previous_direction.set_value(numpy.zeros(self.parameter_updater.param_length, dtype = theano.config.floatX))

                '''
                print(self.parameter_updater.previous_direction.get_value())
                print(numpy.dot(self.parameter_updater.previous_direction.get_value(),self.parameter_updater.previous_direction.get_value()))
                print(self.parameter_updater.previous_gradient.get_value())
                '''
                in_batch_counter += 1
                iteration_counter += 1

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
                if iteration_counter >= 500:
                    for epoch_callback in self.epoch_callbacks:
                        x = epoch_callback.on_epoch()

                    self.epoch_callbacks[-1].callbacks[0](x, None)
                    iteration_counter = 0

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise