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


class Bgfs(object):


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

        '''
        self.full_training_iterator = training_set.iterator(iterator_type='sequential',
                                                        loop_style='divisible',
                                                        batch_size=10000)

        self.training_iterator2 = training_set.iterator(iterator_type='sequential',
                                                        loop_style='divisible',
                                                        batch_size=1000)
        '''

        self.loss_function = theano.function(input_symbols,scalar_loss)

        self.new_epoch = True
#        self.method = self._parameter_updaters[0].method

        self.param_shapes = param_shapes
        # Initialize saved variables:
        self.y = []
        self.s = []
        self.rho = []
        self.grad_log = []
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
            self.validation_function_value = self.get_validation_function_value()
            self.validation_function_value_log.append(self.validation_function_value)	


    '''
    def get_full_gradient(self):

        full_gradient = 0

        for _ in range(self.batches_in_epoch_full_gradient):
            cost_arguments = self.full_training_iterator.next()
            full_gradient = full_gradient + (self.gradient_function(*cost_arguments) / self.batches_in_epoch_full_gradient)

        return full_gradient

    def get_gradient(self):

        cost_args = self.training_iterator.next()
        return self.gradient_function(*cost_args)
    '''

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

        print(validation_function_value)
        return validation_function_value


    def LBFGS_step(self):

        # Initialize parameters for backtracking:
        tau = 0.5
        beta_ = 1E-4
        tolerance = 1E-6
        learning_rate = self.learning_rate
        alpha_found = False

        '''
        grad = self.get_gradient()

        if self.k > 0:
            self.y.append(grad - self.grad_log[-1])
            temp = numpy.dot(self.y[-1],self.s[-1])
            new_rho = 1.0/temp
            self.rho.append(new_rho)
            validation_function_value_previous = self.validation_function_value_log[-1]
        else:
            self.s.append(self.parameters.get_value())
            self.y.append(grad)
            temp = numpy.dot(self.y[-1],self.s[-1])
            new_rho = 1.0/temp
            self.rho.append(new_rho)
            validation_function_value_previous = self.get_validation_function_value()
            self.validation_function_value_log.append(validation_function_value_previous)

        self.grad_log.append(grad)

        current_parameters = self.parameters.get_value()
        n = current_parameters.shape[0]
        m = 10

        alpha = [None]*min(self.k,m)
        beta = [None]*min(self.k,m)
        r = [None]*min(self.k+1,m+1)

        q = grad
        for i in list(reversed(range(0, min(m,self.k)))):
            alpha[i] = self.rho[i] * numpy.dot(numpy.transpose(self.s[i]),q)
            q = q - alpha[i]*self.y[i]

        if self.k > 0:
            # numpy.identity(n)
            H0 =  ( ( numpy.dot( numpy.transpose(self.y[self.k-1]),self.s[self.k-1] ) )/( numpy.dot( numpy.transpose(self.y[self.k-1]),self.y[self.k-1] ) ) )
            r[0] = H0*q
        else:
            #H0 = numpy.identity(n)
            r[0] = q


        for i in range(min(m,self.k)):
            beta[i] = self.rho[i] * numpy.dot(numpy.transpose(self.y[i]),r[i])
            r[i+1] = r[i] + numpy.dot( self.s[i], alpha[i] - beta[i] )

        if self.k > 0:
            direction = r[min(m,self.k)]
        else:
            direction = r[0]


        q = grad
        alpha = [None]*min(self.k,m)

        for i in range(min(self.k,m)):
            alpha[i] = self.rho[self.k-i]*numpy.dot(self.s[self.k-i],q)
            q = q - alpha[i]*self.y[self.k-i]

        if self.k > 0:
            # numpy.identity(n)
            H0 =  numpy.dot(self.y[self.k-1],self.s[self.k-1]) / numpy.dot(self.y[self.k-1],self.y[self.k-1])
            r = H0*q
        else:
            r = q

        for i in list(reversed(range(min(self.k, m)))):
            beta = self.rho[self.k-i]*numpy.dot(self.y[self.k-i],r)
            r = r + self.s[self.k-i]*(alpha[i] - beta)

        direction = r
        '''

        grad = self.get_gradient2()

        if self.k > 0:
            self.y.append(grad - self.grad_log[-1])
            temp = numpy.dot(self.y[-1],self.s[-1])
            new_rho = 1.0/temp
            self.rho.append(new_rho)
        else:
            print("L-BFGS is now carried out.")

        if self.armijo == True:
            validation_function_value_previous = self.validation_function_value_log[-1]

        self.grad_log.append(grad)

        current_parameters = self.parameters.get_value()
        n = current_parameters.shape[0]
        m = 20

        q = grad
        alpha = [None]*(min(self.k,m)+1)

        for i in range(1, (min(self.k,m)+1)):
            alpha[i] = self.rho[self.k-i]*numpy.dot(self.s[self.k-i],q)
            q = q - alpha[i]*self.y[self.k-i]

        if self.k > 0:
            # numpy.identity(n)
            H0 =  numpy.dot(self.y[self.k-1],self.s[self.k-1]) / numpy.dot(self.y[self.k-1],self.y[self.k-1])
            r = H0*q
        else:
            r = q

        for i in list(reversed(range(1, (min(self.k, m)+1)))):
            beta = self.rho[self.k-i]*numpy.dot(self.y[self.k-i],r)
            r = r + self.s[self.k-i]*(alpha[i] - beta)

        direction = r

        if self.armijo == True: # Perform line search

            inner_loop_counter = 0
            while not alpha_found:

                new_params = current_parameters - learning_rate * direction

                # Restrict norms of the columns to be unit norms
                '''
                index_from = 0
                for shape in self.param_shapes:
                    if len(shape) == 4: # it is a filter
                        col_length = shape[2]*shape[3]
                        for _ in range(shape[0]*shape[1]):
                            index_to = index_from + col_length
                            new_params[index_from:index_to] = new_params[index_from:index_to]/numpy.linalg.norm(new_params[index_from:index_to])
                            index_from = index_to
                    else:
                        index_from += numpy.product(shape)
                '''

                if self.tangent:
                    # Use tangent approach:
                    index_from = 0
                    for shape in self.param_shapes:
                        if len(shape) == 4: # it is a filter
                            col_length = shape[2]*shape[3]
                            for _ in range(shape[0]*shape[1]):
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


                '''
                print(direction)
                print(direction.shape)
                print(new_params)
                print(new_params.shape)
                '''
                new_params = new_params.astype(theano.config.floatX)
                #direction_ = new_params - current_parameters

                # Take step:
                self.parameters.set_value(new_params)

                validation_function_value = self.get_validation_function_value()

                val2 = validation_function_value_previous - (beta_ * learning_rate * numpy.dot(grad,direction))
                # Do the actual backtracking:
                if validation_function_value < val2+tolerance or inner_loop_counter > 20:
                    alpha_found = True
                else:
                    learning_rate = tau*learning_rate
                    inner_loop_counter += 1

            # Learning rate FOUND
            # Learning rate is found
            print("learning rate used: ", learning_rate)
            self.validation_function_value_log.append(validation_function_value)

        else:
            new_params = current_parameters - learning_rate * direction
            if not self.tangent:
                new_params = new_params.astype(theano.config.floatX)
                self.parameters.set_value(new_params)

        if self.tangent == True and self.armijo == False:
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

        '''
        if self.armijo == True:
            # Save validation function value for next time to save time
            if self.tangent:
                validation_function_value = self.get_validation_function_value()
                self.validation_function_value_log.append(validation_function_value)

            # Learning rate is found
            print("learning rate used: ", learning_rate)
        '''

        # Update lists:
        self.s.append(new_params - current_parameters)

        self.k = self.k + 1 # Update counter

        '''
        #Try normal SGD:
        grad = self.get_gradient(cost_arguments)
        parameter_value = self.parameters.get_value()
        new_param = parameter_value - self.learning_rate * grad
        self.parameters.set_value(new_param)
        '''

    def SGD_step_armijo(self):

        tau = 0.5
        beta_ = 1E-4
        #tolerance = 1E-6
        tolerance = 0.01
        alpha_found = False
        validation_function_value_previous = self.validation_function_value

        learning_rate = self.learning_rate
        grad = self.get_gradient2()

        current_parameters = self.parameters.get_value()

        in_batch_counter = 0
        while not alpha_found:

            direction = - learning_rate * grad
            new_params = current_parameters + direction
            new_params = new_params.astype(theano.config.floatX)
            self.parameters.set_value(new_params)

            self.validation_function_value = self.get_validation_function_value()

            val2 = validation_function_value_previous - (beta_ * learning_rate * numpy.dot(grad,direction))
            # Do the actual backtracking:
            if self.validation_function_value < val2+tolerance or in_batch_counter > 10:
                alpha_found = True
                print "Alpha used:", learning_rate
            else:
                learning_rate = tau*learning_rate
                in_batch_counter += 1


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
                    self.SGD_step_armijo()

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
