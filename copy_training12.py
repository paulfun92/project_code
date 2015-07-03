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
from copy_dataset import DataIterator
from simplelearn.utils import safe_izip
from simplelearn.formats import Format
from copy_nodes import Node
import pdb

# pylint: disable=too-few-public-methods


class StopTraining(Exception):
    '''
    An exception thrown to signal the end of training.

    Analogous to the built-in exception StopIteration.
    '''
    def __init__(self, status, message):
        if status not in ('ok', 'error'):
            raise ValueError("Expected StopTraining status to be 'ok' or "
                             "'error', but got '%s'." % str(status))

        self.status = status
        super(StopTraining, self).__init__(message)


class EpochCallback(object):
    '''
    Abstract class for callbacks to call between training epochs.
    '''

    def on_start_training(self):
        '''
        Called at the beginning of training, before processing any batches.
        '''
        raise NotImplementedError("%s.on_start_training() not yet implemented."
                                  % type(self))

    def on_epoch(self):
        '''
        Called after each epoch of training.
        '''
        raise NotImplementedError("%s.on_epoch() not yet implemented." %
                                  type(self))


class LimitsNumEpochs(EpochCallback):
    '''
    Throws a StopTraining exception after a fixed number of epochs.
    '''

    def __init__(self, max_num_epochs):
        if not numpy.issubdtype(type(max_num_epochs), numpy.integer):
            raise TypeError("Expected max_num_epochs to be an integer, not a "
                            "%s." % type(max_num_epochs))

        if max_num_epochs < 0:
            raise ValueError("max_num_epochs must be non-negative, got %d." %
                             max_num_epochs)

        self._max_num_epochs = max_num_epochs
        self._num_epochs_seen = None

    def on_start_training(self):
        self._num_epochs_seen = 0

    def on_epoch(self):
        assert self._num_epochs_seen >= 0

        self._num_epochs_seen += 1

        if self._num_epochs_seen >= self._max_num_epochs:
            raise StopTraining(status='ok',
                               message=('Reached max # of epochs %d.' %
                                        self._max_num_epochs))

class PicklesOnEpoch(EpochCallback):
    '''
    A callback that saves a list of objects at the start of training, and
    again on each epoch.
    '''
    def __init__(self, objects, filepath, overwrite=True):
        '''
        Parameters
        ----------
        objects: OrderedDict
          Maps names to picklable objects. This dict is pickled at each epoch.
          Note that dynamically created functions (e.g. inner functions that
          aren't publically accessible by name) are not picklable.
          module-level An object, or a sequence of objects, to pickle at each
          epoch.

        filepath: str
          The file path to save the objects to. Must end in '.pkl'

        overwrite: bool
          Overwrite the file at each epoch. If False, and filepath is
          'path/to/file.pkl', this saves a separate file for each epoch,
          of the form 'path/to/file_00000.pkl', 'path/to/file_00001.pkl',
          etc. The first file stores the state of <objects> before any epochs.
        '''
        assert_is_instance(objects, OrderedDict)
        for key in objects.keys():
            assert_is_instance(key, basestring)

        if os.path.isdir(filepath):
            path = filepath
            filename = ""
        else:
            path, filename = os.path.split(filepath)

        assert_true(os.path.isdir(path), "{} isn't a directory".format(path))
        assert_equal(os.path.splitext(filename)[1], '.pkl')


        self._objects_to_pickle = objects
        self._filepath = filepath
        self._overwrite = overwrite
        self._num_epochs_seen = 0

    def on_start_training(self):
        self.on_epoch()

    def on_epoch(self):
        if self._overwrite:
            filepath = self._filepath
        else:
            path, filename = os.path.split(self._filepath)
            extension = os.path.splitext(filename)[1]
            filename = '%s_%05d%s' % (filename,
                                      self._num_epochs_seen,
                                      extension)

            filepath = os.path.join(path, filename)

        with file(filepath, 'wb') as pickle_file:

            cPickle.dump(self._objects_to_pickle,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

        self._num_epochs_seen += 1

class ValidationCallback(EpochCallback):
    '''
    Evaluates some Monitors over validation data in between training epochs.
    '''

    def __init__(self, inputs, input_iterator, monitors):
        '''
        Parameters
        ----------

        inputs: sequence of theano.gof.Variables
          Symbols for the outputs of the input_iterator.

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of validation set batches, such as (values, labels).

        monitors: sequence of Monitors.
          These are also used as epoch callbacks.
        '''

        #
        # Checks inputs
        #

        assert_is_instance(inputs, Sequence)
        for input_symbol in inputs:
            assert_is_instance(input_symbol, theano.gof.Variable)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        assert_is_instance(monitors, Sequence)
        assert_greater(len(monitors), 0)

        for monitor in monitors:
            assert_is_instance(monitor, Monitor)

        #
        # Sets members
        #

        self._input_iterator = input_iterator

        outputs = []
        for monitor in monitors:
            outputs.extend(monitor.monitored_values)

        self._monitors = monitors

        self._update_function = theano.function(inputs, outputs)

    def on_start_training(self):
        classification_error = self.on_epoch()
        return classification_error

    def on_epoch(self):
        '''
        Loops through an epoch of the validation dataset.
        '''

        # Calls monitors' on_start_training()
        for monitor in self._monitors:
            monitor.on_start_training()

        # Repeatedly calls monitors' on_batch()
        keep_going = True

        while keep_going:
            input_batches = self._input_iterator.next()
            keep_going = not self._input_iterator.next_is_new_epoch()

            # pylint: disable=star-args
            outputs = self._update_function(*input_batches)

            output_index = 0
            for monitor in self._monitors:
                new_output_index = (output_index +
                                    len(monitor.monitored_values))
                assert_less_equal(new_output_index, len(outputs))
                monitored_values = outputs[output_index:new_output_index]

                monitor.on_batch(input_batches, monitored_values)

                output_index = new_output_index

        # Calls monitors' on_epoch() methods.
        for monitor in self._monitors:
            classification_error = monitor.on_epoch()

        return classification_error

class LinearlyInterpolatesOverEpochs(EpochCallback):
    '''
    Linearly interpolates a theano shared variable over epochs.
    '''

    def __init__(self,
                 shared_value,
                 final_value,
                 epochs_to_saturation):
        assert_is_instance(shared_value,
                           theano.tensor.sharedvar.SharedVariable)
        assert_is_subdtype(shared_value.dtype, numpy.floating)

        assert_equal(shared_value.ndim == 0, numpy.isscalar(final_value))

        if numpy.isscalar(final_value):
            assert_floating(final_value)
        else:
            assert_is_subdtype(final_value.dtype, numpy.floating)
            assert_equal(final_value.shape,
                         shared_value.get_value().shape)

        assert_integer(epochs_to_saturation)
        assert_greater(epochs_to_saturation, 0)

        self.shared_value = shared_value

        cast = numpy.cast[shared_value.dtype]
        self._final_value = cast(final_value)

        self._epochs_to_saturation = epochs_to_saturation

        self._num_epochs_seen = None
        self._initial_value = None

    def on_start_training(self):
        self._num_epochs_seen = 0
        self._initial_value = self.shared_value.get_value()

    def on_epoch(self):
        assert_greater_equal(self._num_epochs_seen, 0)
        self._num_epochs_seen += 1

        cast = numpy.cast[self.shared_value.dtype]

        # interpolation parameter
        end_weight = cast(min(
            1.0,
            float(self._num_epochs_seen) / self._epochs_to_saturation))

        start_weight = cast(1.0) - end_weight

        self.shared_value.set_value(
            start_weight * self._initial_value +
            end_weight * self._final_value)


class LinearlyScalesOverEpochs(EpochCallback):
    '''
    Linearly scales a theano shared variable over epochs.

    Parameters
    ----------

    shared_value: a Theano shared variable
      This value will be scaled in-place by a factor S that decreases from 1.0
      to final_scale over <epochs_to_saturation> epochs.

    final_scale: float
      Final value of S. Mutually exclusive with final_value.

    final_value: numpy.ndarray
      A numpy array of the same shape as shared_value.get_value().shape.
      Mutually exclusive with final_scale.

    epochs_to_saturation: int
      self._scale should decay to final_value after this many epochs.
    '''

    def __init__(self,
                 shared_value,
                 final_scale,
                 epochs_to_saturation):
        assert_is_instance(shared_value,
                           theano.tensor.sharedvar.SharedVariable)
        assert_floating(final_scale)
        assert_greater_equal(final_scale, 0.0)
        assert_less_equal(final_scale, 1.0)
        assert_integer(epochs_to_saturation)
        assert_greater(epochs_to_saturation, 0)

        self.shared_value = shared_value
        self._final_scale = final_scale
        self._epochs_to_saturation = epochs_to_saturation

        # initialized in on_start_training()
        self._initial_value = None
        self._num_epochs_seen = None

    def on_start_training(self):
        self._num_epochs_seen = 0
        self._initial_value = self.shared_value.get_value()

    def on_epoch(self):
        assert_greater_equal(self._num_epochs_seen, 0)
        self._num_epochs_seen += 1

        # interpolation parameter
        alpha = min(1.0,
                    float(self._num_epochs_seen) / self._epochs_to_saturation)

        scale = (1.0 - alpha) + alpha * self._final_scale

        self.shared_value.set_value(scale * self._initial_value)


class Monitor(EpochCallback):
    '''
    On each epoch, this reports statistics about some monitored value Y.

    Examples: Y might be the output of layer 3 of a 6-layer net.

              MaxMonitor reports the elementwise maximum of Y encountered
              over the epoch.

              AverageMonitor reports Y, elementwise-averaged over the epoch.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        '''
        Parameters
        ----------
        values_to_monitor: theano expression, or a Sequence of them
          A sequence of theano expressions to monitor. These should be
          functions of the input variables.

          Must not be empty.

        formats: a Format, or a Sequence of them
          A sequence of the above values' Formats.

        callbacks: a __call__-able, or a Sequence of them
          The values returned by self._on_epoch() get fed to these callbacks.
          These must have the call signature f(values, formats).
          Values is the Sequence returned by self._on_epoch().
          Formats are the values' formats, also a Sequence.
        '''

        #
        # Checks args
        #

        if isinstance(values_to_monitor, theano.gof.Variable):
            values_to_monitor = [values_to_monitor]

        if isinstance(formats, Format):
            formats = [formats]

        if not isinstance(callbacks, Sequence):
            callbacks = [callbacks]

        assert_is_instance(values_to_monitor, Sequence)
        assert_is_instance(formats, Sequence)
        assert_is_instance(callbacks, Sequence)

        assert_equal(len(values_to_monitor), len(formats))
        assert_equal(len(values_to_monitor),
                     len(frozenset(values_to_monitor)),
                     "values_to_monitor contains repeated elements: %s" %
                     str(values_to_monitor))

        for value, fmt in safe_izip(values_to_monitor, formats):
            assert_is_instance(value, theano.gof.Variable)
            assert_is_instance(fmt, Format)

        #
        # Sets members
        #

        self.monitored_values = tuple(values_to_monitor)
        self._formats = tuple(formats)
        self._callbacks = list(callbacks)

    def on_batch(self, input_batches, monitored_value_batches):
        '''
        Updates the values to report at the end of the epoch.

        Parameters
        ----------
        input_batches: Sequence of numpy.ndarrays
          The input batches coming in from the dataset's iterator.
          Typically these are (values, labels)

        monitored_value_batches: Sequence of numpy.ndarrays
          The numerical values, for this batch, of the values_to_monitor
          arguments to __init__().
        '''
        assert_equal(len(monitored_value_batches), len(self._formats))

        for batch, fmt in safe_izip(monitored_value_batches, self._formats):
            fmt.check(batch)

        self._on_batch(tuple(input_batches),
                       tuple(monitored_value_batches))

    def _on_batch(self, input_batches, monitored_value_batches):
        '''
        Implementation of self.on_batch(). See that method's docs.

        Parameters
        ----------
        input_batches: tuple of numpy.ndarrays.

        monitored_value_batches: tuple of numpy.ndarrays.
        '''
        raise NotImplementedError("%s._on_batch() not yet implemented." %
                                  type(self))

    def on_start_training(self):
        pass

    def on_epoch(self):
        '''
        Feeds monitored values to self._callbacks
        '''
        # compute values to report
        values_to_report = self._on_epoch()

        if not isinstance(values_to_report, tuple):
            raise ValueError("%s._on_epoch() implemented incorrectly. It "
                             "should return a tuple, but it returned %s."
                             % (type(self), type(values_to_report)))

        for callback in self._callbacks:
            callback(values_to_report, self._formats)

        return values_to_report

    def _on_epoch(self):
        '''
        Returns a tuple of values to feed to self._callbacks as arguments.

        Returns
        -------
        rval: tuple of numpy.ndarrays
           Arguments to feed to self._callbacks' __call__(self, *args)
        '''
        raise NotImplementedError("%s._on_epoch() not yet implemented" %
                                  type(self))


class ReduceMonitor(Monitor):
    '''
    An abstract superclass of monitors like MaxMonitor, MinMonitor,
    that operate by applying a reduction operator (e.g. max, min)
    along the batch axis for each batch.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(ReduceMonitor, self).__init__(values_to_monitor,
                                            formats,
                                            callbacks)

        assert_greater(len(self._formats), 0)
        assert_greater(len(self._callbacks), 0)

        for fmt in self._formats:
            assert_in('b', fmt.axes)

        self._tallies = None

    def on_start_training(self):
        self._tallies = None

    def _reduce_batch(self, input_batch, batch_axis):
        '''
        Reduce input_batch along its batch_axis, and return the result.
        '''
        raise NotImplementedError("%s._reduce_batch() not yet implemented." %
                                  type(self))

    def _update_tally(self, reduced_value, batch_axis, tally):
        '''
        Updates a tally (one of self._tallies) using a reduced batch.
        '''
        raise NotImplementedError("%s._update_tally() not yet implemented." %
                                  type(self))

    def _on_batch(self, input_batches, monitored_value_batches):
        batch_axes = [fmt.axes.index('b') for fmt in self._formats]

        new_tallies = []
        for batch, fmt, batch_axis in safe_izip(monitored_value_batches,
                                                self._formats,
                                                batch_axes):
            new_tally = self._reduce_batch(batch, batch_axis)
            fmt.check(new_tally)
            assert_equal(new_tally.shape[batch_axis], 1)

            new_tallies.append(new_tally)

        new_tallies = tuple(new_tallies)

        if self._tallies is None:
            self._tallies = new_tallies
        else:
            for new_tally, old_tally, batch_axis in safe_izip(new_tallies,
                                                              self._tallies,
                                                              batch_axes):
                self._update_tally(new_tally, batch_axis, old_tally)

    def _on_epoch(self):
        assert_is_not(self._tallies, None)

        result = self._tallies
        self._tallies = None

        return result


class MaxMonitor(ReduceMonitor):
    '''
    Computes the elementwise maximum of monitored values, over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(MaxMonitor, self).__init__(values_to_monitor, formats, callbacks)

    def _reduce_batch(self, input_batch, batch_axis):
        return numpy.max(input_batch, axis=batch_axis)

    def _update_tally(self, reduced_value, batch_axis, tally):
        stack = numpy.concatenate((reduced_value, tally), axis=batch_axis)
        tally[...] = numpy.max(stack, axis=batch_axis, keepdims=True)


class MinMonitor(ReduceMonitor):
    '''
    Computes the elementwise minimum of monitored values, over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(MinMonitor, self).__init__(values_to_monitor, formats, callbacks)

    def _reduce_batch(self, input_batch, batch_axis):
        return numpy.min(input_batch, axis=batch_axis)

    def _update_tally(self, reduced_value, batch_axis, tally):
        stack = numpy.concatenate((reduced_value, tally), axis=batch_axis)
        tally[...] = numpy.min(stack, axis=batch_axis, keepdims=True)


class SumMonitor(ReduceMonitor):
    '''
    Computes the elementwise sum of monitored values over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        if not isinstance(formats, Sequence):
            formats = [formats]
            assert not isinstance(values_to_monitor, Sequence)
            values_to_monitor = [values_to_monitor]

        # _reduce_batch() upgrades small int dtypes (e.g. uint8) to larger int
        # dtypes to avoid over/underflow when summing large numbers of them.
        # We need to make their corresponding formats agnostic to dtype, so
        # that they don't raise a stink about batch/tally dtypes being
        # different from the format's expected dtype.
        def remove_small_int_dtype(fmt):
            '''
            Return a copy of fmt, with dtype=None if orig. dtype was small int.
            '''
            if fmt.dtype is not None and numpy.issubdtype(fmt.dtype,
                                                          numpy.integer):
                result = copy.deepcopy(fmt)
                result.dtype = None
                return result
            else:
                return fmt

        formats = [remove_small_int_dtype(fmt) for fmt in formats]

        super(SumMonitor, self).__init__(values_to_monitor,
                                         formats,
                                         callbacks)
        self._count = None

    def _reduce_batch(self, input_batch, batch_axis):

        def upcast_if_integer(input_batch):
            '''
            Cast to int64 iff input_batch.dtype is an integral dtype.

            Lowers the risk of integer over/underflow (esp. if dtype is uint8).
            '''
            if numpy.issubdtype(input_batch.dtype, numpy.integer):
                return numpy.cast['int64'](input_batch)
            else:
                return input_batch

        return numpy.sum(upcast_if_integer(input_batch),
                         axis=batch_axis,
                         keepdims=True)

    def _update_tally(self, reduced_value, batch_axis, tally):
        tally += reduced_value


class AverageMonitor(SumMonitor):
    '''
    Computes the elementwise average of monitored values over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(AverageMonitor, self).__init__(values_to_monitor,
                                             formats,
                                             callbacks)
        self._count = 0

    def _on_batch(self, input_batches, monitored_value_batches):
        # Update self._tallies
        super(AverageMonitor, self)._on_batch(input_batches,
                                              monitored_value_batches)
        assert_is_instance(self._tallies, Sequence)

        batch_axes = [fmt.axes.index('b') for fmt in self._formats]

        # Update self._count
        batch_sizes = numpy.asarray([batch.shape[batch_axis]
                                     for batch, batch_axis
                                     in safe_izip(monitored_value_batches,
                                                  batch_axes)])
        assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]),
                    "Unequal batch sizes: %s" % str(batch_sizes))
        self._count += batch_sizes[0]

    def _on_epoch(self):
        totals = super(AverageMonitor, self)._on_epoch()
        assert_is_instance(totals, Sequence)

        result = tuple(total / float(self._count) for total in totals)
        self._count = 0

        return result


class SavesAtMinimum(object):
    '''
    A callback to Monitor that pickles an object (typically the model)
    when some monitored scalar value hits an all-time low.
    '''

    def __init__(self, object_to_save, output_filepath):
        '''
        Parameters
        ----------
        object_to_save: A picklable object

        output_filepath: string
          The file path to save object_to_save to.
        '''
        assert_true(os.path.isdir(os.path.dirname(output_filepath)))

        self._object_to_save = object_to_save
        self._output_filepath = output_filepath
        self._min_value = None


    def __call__(self, values, formats):
        assert_equal(len(values), 1)
        assert_equal(len(values), len(formats))

        fmt = formats[0]
        assert_equal(fmt.axes, ('b', ))

        assert_equal(values[0].shape, (1, ))
        value = values[0][0]

        old_min_value = self._min_value

        if self._min_value is None or value < self._min_value:
            self._min_value = value

        if old_min_value != self._min_value:
            pickle_file = file(self._output_filepath, 'wb')
            cPickle.dump(self._object_to_save,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

class StopsOnStagnation(object):
    '''
    A callback to Monitor that stops training if the monitored value
    (e.g. average loss over the epoch) doesn't decrease for N epochs.
    '''

    def __init__(self, max_epochs, min_proportional_decrease=0.0):
        '''
        max_epochs: int
          Wait for max this many epochs for the monitored value to decrease.

        min_proportional_decrease: float
          If this value is T, the monitored value is V, and the last known
          minimum of V is Vm, then V is considered a decrease only if
          V < (1.0 - T) * Vm
        '''
        assert_greater(max_epochs, 0)
        assert_true(numpy.issubdtype(type(max_epochs), numpy.integer))

        assert_greater_equal(min_proportional_decrease, 0.0)

        self._max_epochs_since_min = max_epochs
        self._min_proportional_decrease = min_proportional_decrease
        self._epochs_since_min = 0
        self._min_value = None

    def __call__(self, values, formats):
        assert_equal(len(values), 1)
        assert_equal(len(values), len(formats))

        fmt = formats[0]
        assert_equal(fmt.axes, ('b', ))

        assert_equal(values[0].shape, (1, ))
        value = values[0][0]

        if self._min_value is None:
            self._min_value = value
        elif value < (1.0 - self._min_proportional_decrease) * self._min_value:
            self._epochs_since_min = 0
            self._min_value = value
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min >= self._max_epochs_since_min:
            message = ("%s stopping training. Value did not lower %s"
                       "below last min value of %g for %d epochs." %
                       (type(self),
                        ("more than %g " % self._min_proportional_decrease
                         if self._min_proportional_decrease > 0.0
                         else ""),
                        self._min_value,
                        self._epochs_since_min))

            raise StopTraining("ok", message)


class LogsToLists(object):
    '''
    A callback to Monitor that logs monitored values to lists.
    '''
    def __init__(self):
        self.logs = None

    def __call__(self, values, formats):
        assert_equal(len(values), len(formats))
        assert_greater(len(values), 0)

        if self.logs is None:
            self.logs = [list() for value in values]
        else:
            assert_equal(len(self.logs), len(values))

        for log, value in safe_izip(self.logs, values):
            log.append(value)



class Sgd(object):

    def __init__(self,
                 inputs,
                 input_iterator,
                 parameters,
                 monitors,
                 epoch_callbacks,
                 training_set,
                 gradient,
                 learning_rate,
                 momentum,
                 loss_function):

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
        self._monitors = tuple(monitors)

        self.input_symbols = [i.output_symbol for i in inputs]

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

        floatX = theano.config.floatX

        def concat(str0, str1):
            '''
            Like str0 + str1, except returns None if either is None.
            '''
            if str0 is None or str1 is None:
                return None
            else:
                return str0 + str1

        def make_shared_floatX(numeric_var, name, **kwargs):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),name=name,**kwargs)

        # self._training_set = training_set
        self._training_full_gradient_iterator = training_set.iterator(iterator_type='sequential', batch_size=50000) #training_set.size
        self.classification_errors = numpy.asarray([])

        self.gradient = gradient
        self.loss_function = loss_function

        self.gradient_function = theano.function(self.input_symbols,self.gradient)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_found = False

        self.velocity = []
        for parameter in self._parameters:
            self.velocity.append(make_shared_floatX(0.0 * parameter.get_value(), concat(parameter.name, ' velocity'), broadcastable=parameter.broadcastable))

        param_symbols = list(self._parameters)
        desired_param_symbols = []
        for i in range(len(self._parameters)/2):
            desired_param_symbols.append(theano.tensor.matrix(dtype=theano.config.floatX))
            desired_param_symbols.append(theano.tensor.vector(dtype=theano.config.floatX))

        self.objective_value_given_parameters = theano.function([input_symbol for input_symbol in self.input_symbols] + [desired_param_symbol for desired_param_symbol in desired_param_symbols], loss_function, givens=[ (param_symbol, desired_param_symbol) for param_symbol, desired_param_symbol in safe_izip(param_symbols, desired_param_symbols) ] )
        self.objective_value = theano.function(self.input_symbols, loss_function)



    def get_gradient(self, cost_arguments):

        return self.gradient_function(*cost_arguments)
        # Function that returns gradient of the loss function w.r.t the parameters W and b,
        # at point of current parameters, given the input batch of images.


    def sgd_step(self, cost_arguments):

        grad = self.get_gradient(cost_arguments)
        learning_rate_found = False
        learning_rate = self.learning_rate
        counter = 0
        cost_arguments_full = self._training_full_gradient_iterator.next()

        while not learning_rate_found:

            counter = counter + 1
            new_params = []
            new_velocities = []

            for i in range(len(self._parameters)):  # Update every parameters seperately in this loop
                velocity = self.velocity[i].get_value()
                parameter_value = self._parameters[i].get_value()

                new_velocity = self.momentum*velocity - learning_rate * grad[i]
                new_velocities.append(new_velocity)

                new_param = parameter_value + new_velocity
                new_params.append(new_param)

            arguments = list(cost_arguments_full) + new_params
            new_objective_value = self.objective_value_given_parameters(*arguments)
            current_objective_value = self.objective_value(*cost_arguments_full)

            if new_objective_value <= current_objective_value:
                print("Learning rate:", learning_rate)
                print("Objective value:", new_objective_value)
                learning_rate_found = True
                for i in range(len(self._parameters)):
                    self._parameters[i].set_value(new_params[i])
                    self.velocity[i].set_value(new_velocities[i])
            elif counter>10:
                learning_rate = 0.01
                print("Learning rate:", learning_rate, "loop")
                print("Objective value:", new_objective_value)

                for i in range(len(self._parameters)):  # Update every parameters seperately in this loop
                    velocity = self.velocity[i].get_value()
                    parameter_value = self._parameters[i].get_value()

                    new_velocity = self.momentum*velocity - learning_rate * grad[i]
                    new_param = parameter_value + new_velocity

                    self._parameters[i].set_value(new_velocity)
                    self.velocity[i].set_value(new_param)

                learning_rate_found = True

            else:
                learning_rate = 0.5 * learning_rate


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
                classification_error = callback.on_start_training()
                if classification_error is not None:
                    self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

            while True:

                # First, calculate full gradient (at the beginning of each epoch)
                # gets batch of data
                # batch_size_full_gradient

                cost_arguments = self._input_iterator.next()
                self.sgd_step(cost_arguments)


                # calls epoch callbacks, if we've iterated through an epoch
                if self._input_iterator.next_is_new_epoch() or self.min_found==True:
                    for callback in all_callbacks:
                        classification_error = callback.on_epoch()
                        if classification_error is not None:
                            self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

                if self.min_found==True:

                    raise StopTraining(status='ok', message=('Reached minimum'))

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise


    # def __getstate__(self):
    #     result = dict()
    #     result.update(self.__dict__)
    #     result['_update_function'] = "left unserialized"
    #     return result

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     assert_equal(self._update_function, "left unserialized")
    #     self._update_function = self._compile_update_function(
    #         **self._compile_update_function_args)


class SemiSgd(object):

    '''
    Uses semi stochastic gradient descent to optimize a cost w.r.t. parameters.

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
                 monitors,
                 epoch_callbacks,
                 training_set,
                 gradient,
                 learning_rate,
                 momentum,
                 loss_function):

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
        self._monitors = tuple(monitors)

        self.input_symbols = [i.output_symbol for i in inputs]

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

        floatX = theano.config.floatX

        def concat(str0, str1):
            '''
            Like str0 + str1, except returns None if either is None.
            '''
            if str0 is None or str1 is None:
                return None
            else:
                return str0 + str1

        def make_shared_floatX(numeric_var, name, **kwargs):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),name=name,**kwargs)

        # self._training_set = training_set
        self._training_full_gradient_iterator = training_set.iterator(iterator_type='sequential', batch_size=50000) #training_set.size
        self.classification_errors = numpy.asarray([])

        self.gradient = gradient
        self.loss_function = loss_function

        self.gradient_function = theano.function(self.input_symbols,self.gradient)

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocity = []
        for parameter in self._parameters:
            self.velocity.append(make_shared_floatX(0.0 * parameter.get_value(), concat(parameter.name, ' velocity'), broadcastable=parameter.broadcastable))



    def get_gradient(self, cost_arguments):

        return self.gradient_function(*cost_arguments)
        # Function that returns gradient of the loss function w.r.t the parameters W and b,
        # at point of current parameters, given the input batch of images.


    def semi_sgd_step(self, cost_arguments, gradients, full_gradient):

        grad = self.get_gradient(cost_arguments)

        for i in range(len(self._parameters)):  # Update every parameters seperately in this loop
            velocity = self.velocity[i].get_value()
            parameter_value = self._parameters[i].get_value()

            new_velocity = self.momentum*velocity - self.learning_rate * (full_gradient[i] + grad[i] - gradients[i])

            new_param = parameter_value + new_velocity
            self._parameters[i].set_value(new_param)
            self.velocity[i].set_value(new_velocity)


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
                classification_error = callback.on_start_training()
                if classification_error is not None:
                    self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

            # Set initial parameters for SemiSGD:
            # max_stochastic_steps_per_epoch = max # of stochastic steps per epoch
            # v = lower bound on the constant of the strongly convex loss function
            # stochastic_steps = # of stochastic steps taken in an epoch, calculated geometrically.
            max_stochastic_steps_per_epoch = 500
            v = 0.05

            while True:

                # First, calculate full gradient (at the beginning of each epoch)
                # gets batch of data
                # batch_size_full_gradient

                # If new epoch is started:
                # if self._training_full_gradient_iterator.next_is_new_epoch(): # Extract
                cost_arguments = self._training_full_gradient_iterator.next()
                full_gradient = self.get_gradient(cost_arguments)

                # Determine # of stochastic steps taken in the epoch:
                # Calculate the sum of the probabilities for geometric distribution:
                sum = 0
                for t in range(1,max_stochastic_steps_per_epoch+1):
                    add = pow((1-v*self.learning_rate),(max_stochastic_steps_per_epoch - t))
                    sum = sum + add

                cummulative_prob = 0
                rand = numpy.random.uniform(0,1)
                for t in range(1,max_stochastic_steps_per_epoch+1):
                    prob = pow((1-v*self.learning_rate),(max_stochastic_steps_per_epoch - t)) / sum
                    cummulative_prob = cummulative_prob + prob
                    if  rand < cummulative_prob:
                        stochastic_steps = t
                        break

                # First calculate all the necessary gradients and cost arguments at the current parameters:
                cost_arguments = []
                gradients_at_params = []
                for i in range(stochastic_steps):
                    cost_arguments.append(self._input_iterator.next())
                    gradients_at_params.append(self.get_gradient(cost_arguments[i]))

                # Run the semi-stochastic gradient descent main loop
                for t in range(stochastic_steps):
                    # Now take a step:
                    self.semi_sgd_step(cost_arguments[t], gradients_at_params[t], full_gradient)


                # calls epoch callbacks, if we've iterated through an epoch
                # if self._input_iterator.next_is_new_epoch():
                for callback in all_callbacks:
                    classification_error = callback.on_epoch()
                    if classification_error is not None:
                        self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise


    # def __getstate__(self):
    #     result = dict()
    #     result.update(self.__dict__)
    #     result['_update_function'] = "left unserialized"
    #     return result

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     assert_equal(self._update_function, "left unserialized")
    #     self._update_function = self._compile_update_function(
    #         **self._compile_update_function_args)


class SemiSgdPlus(object):

    '''
    Uses semi stochastic gradient descent to optimize a cost w.r.t. parameters.

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
                 monitors,
                 epoch_callbacks,
                 training_set,
                 gradient,
                 learning_rate,
                 momentum,
                 loss_function):

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
        self._monitors = tuple(monitors)

        self.input_symbols = [i.output_symbol for i in inputs]

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

        floatX = theano.config.floatX

        def concat(str0, str1):
            '''
            Like str0 + str1, except returns None if either is None.
            '''
            if str0 is None or str1 is None:
                return None
            else:
                return str0 + str1

        def make_shared_floatX(numeric_var, name, **kwargs):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),name=name,**kwargs)

        # self._training_set = training_set
        self._training_full_gradient_iterator = training_set.iterator(iterator_type='sequential', batch_size=50000) #training_set.size
        self.classification_errors = numpy.asarray([])

        self.gradient = gradient
        self.loss_function = loss_function

        self.gradient_function = theano.function(self.input_symbols,self.gradient)

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocity = []
        for parameter in self._parameters:
            self.velocity.append(make_shared_floatX(0.0 * parameter.get_value(), concat(parameter.name, ' velocity'), broadcastable=parameter.broadcastable))



    def get_gradient(self, cost_arguments):

        return self.gradient_function(*cost_arguments)
        # Function that returns gradient of the loss function w.r.t the parameters W and b,
        # at point of current parameters, given the input batch of images.

    def sgd_step(self, cost_arguments):

        grad = self.get_gradient(cost_arguments)

        for i in range(len(self._parameters)):  # Update every parameters seperately in this loop
            velocity = self.velocity[i].get_value()
            parameter_value = self._parameters[i].get_value()

            new_velocity = self.momentum*velocity - self.learning_rate * grad[i]

            new_param = parameter_value + new_velocity
            self._parameters[i].set_value(new_param)
            self.velocity[i].set_value(new_velocity)

    def semi_sgd_step(self, cost_arguments, gradients, full_gradient):

        grad = self.get_gradient(cost_arguments)

        for i in range(len(self._parameters)):  # Update every parameters seperately in this loop
            velocity = self.velocity[i].get_value()
            parameter_value = self._parameters[i].get_value()

            new_velocity = self.momentum*velocity - self.learning_rate * (full_gradient[i] + grad[i] - gradients[i])

            new_param = parameter_value + new_velocity
            self._parameters[i].set_value(new_param)
            self.velocity[i].set_value(new_velocity)


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
                classification_error = callback.on_start_training()
                if classification_error is not None:
                    self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

            # Set initial parameters for SemiSGD:
            # max_stochastic_steps_per_epoch = max # of stochastic steps per epoch
            # v = lower bound on the constant of the strongly convex loss function
            # stochastic_steps = # of stochastic steps taken in an epoch, calculated geometrically.
            max_stochastic_steps_per_epoch = 500
            v = 0.05
            sgd_first_loop_done = False

            while True:

                # First, calculate full gradient (at the beginning of each epoch)
                # gets batch of data
                # batch_size_full_gradient

                while sgd_first_loop_done == False:
                    cost_arguments = self._input_iterator.next()
                    self.sgd_step(cost_arguments)

                    if self._input_iterator.next_is_new_epoch():
                        sgd_first_loop_done = True
                        for callback in all_callbacks:
                            classification_error = callback.on_epoch()
                            if classification_error is not None:
                                self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

                # If new epoch is started:
                # if self._training_full_gradient_iterator.next_is_new_epoch(): # Extract
                cost_arguments = self._training_full_gradient_iterator.next()
                full_gradient = self.get_gradient(cost_arguments)
                stochastic_steps = 500

                # First calculate all the necessary gradients and cost arguments at the current parameters:
                cost_arguments = []
                gradients_at_params = []
                for i in range(stochastic_steps):
                    cost_arguments.append(self._input_iterator.next())
                    gradients_at_params.append(self.get_gradient(cost_arguments[i]))

                # Run the semi-stochastic gradient descent main loop
                for t in range(stochastic_steps):
                    # Now take a step:
                    self.semi_sgd_step(cost_arguments[t], gradients_at_params[t], full_gradient)


                # calls epoch callbacks, if we've iterated through an epoch
                # if self._input_iterator.next_is_new_epoch():
                for callback in all_callbacks:
                    classification_error = callback.on_epoch()
                    if classification_error is not None:
                        self.classification_errors = numpy.append(self.classification_errors,classification_error[0])

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return self.classification_errors
            else:
                raise
