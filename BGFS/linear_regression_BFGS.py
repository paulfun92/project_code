#! /usr/bin/env python

'''
Steps through the iterations of a simple 2D->1D linear regression problem.
'''

import sys
import argparse
import numpy

from matplotlib import pyplot

# This line required for projection='3d' keyword to work
import mpl_toolkits.mplot3d   # pylint: disable=unused-import

from nose.tools import (assert_equal,
                        assert_greater,
                        assert_greater_equal)

import theano
from simplelearn.nodes import AffineTransform, L2Loss
from simplelearn.utils import safe_izip
from simplelearn.data.dataset import Dataset
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  Sgd,
                                  IterationCallback,
                                  LimitsNumEpochs,
                                  LogsToLists,
                                  MeanOverEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation)
import pdb


def parse_args():
    '''
    Parses command-line args and returns them as a namespace.
    '''

    parser = argparse.ArgumentParser(
        description=("Simple demo of stochastic gradient descent, "
                     "with and without Nesterov's accelerated "
                     "gradients."))

    def positive_int(arg):
        '''Arg checker for positive ints.'''
        arg = int(arg)
        assert_greater(arg, 0)

        return arg

    def positive_float(arg):
        '''Arg checker for positive floats.'''
        arg = float(arg)
        assert_greater(arg, 0)

        return arg

    def non_negative_float(arg):
        '''Arg checker for non-negative floats.'''
        arg = float(arg)
        assert_greater_equal(arg, 0)

        return arg

    def batch_size(arg):
        '''Checks batch-size argument'''
        arg = int(arg)
        if arg < 1 and arg != -1:
            raise ValueError("Batch size must be positive, or -1, not %d."
                             % arg)

        return arg

    parser.add_argument("--training-size",
                        type=positive_int,
                        default=100,
                        help=("The number of points in the training set"))

    parser.add_argument("--testing-size",
                        type=positive_int,
                        default=10,
                        help=("The number of points in the testing set."))

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=.1,
                        help="Learning rate for SGD")

    parser.add_argument("--momentum",
                        type=non_negative_float,
                        default=.5,
                        help="Momentum for SGD")

    parser.add_argument("--nesterov",
                        type=bool,
                        default=True,
                        help="Set to True to use Nesterov momentum.")

    parser.add_argument("--batch-size",
                        type=batch_size,
                        default=-1,
                        help=("Batch size. Set to -1 (default) to use entire "
                              "dataset."))

    result = parser.parse_args()
    return result


def main():
    '''
    Entry point of script.
    '''

    args = parse_args()

    # pylint: disable=invalid-name, no-member
    floatX = numpy.dtype(theano.config.floatX)

    def affine_transform(matrix, bias, inputs):
        '''
        Returns dot(inputs, matrix) + bias.
        '''
        assert_equal(matrix.ndim, 2)
        assert_equal(bias.ndim, 1)
        assert_equal(bias.shape[0], matrix.shape[1])
        assert_equal(inputs.ndim, 2)

        result = numpy.dot(inputs, matrix) + bias
        assert_equal(result.shape, (inputs.shape[0], bias.shape[0]))
        return result

    def make_random_dataset(matrix,
                            bias,
                            min_input,
                            max_input,
                            output_variance,
                            rng,
                            num_points,
                            dtype):
        '''
        Returns random inputs X, and targets f(X) + N, where N is noise.

        f is an affine transform dot(X, W) + B
        N is normally distributed noise with a given variance.

        Parameters
        ----------
        matrix: numpy.ndarray
          W in the above equation.
        bias: numpy.ndarray
          B in the above equation.
        min_input: float
          Min value of randomly-generated inputs X
        max_input: float
          Min value of randomly-generated inputs X
        output_variance: float
          Variance of N
        rng: numpy.random.RandomState
          Used to generate X, N
        num_points: int
          Number of rows in X
        dtype: numpy.dtype
          dtype of X and targets.
        '''
        assert_greater_equal(output_variance, 0.0)

        num_dims = matrix.shape[0]
        assert_equal(len(min_input), num_dims)
        assert_equal(len(max_input), num_dims)

        inputs = numpy.vstack([rng.uniform(low=min_input[i],
                                           high=max_input[i],
                                           size=num_points)
                               for i in range(num_dims)]).T

        outputs = affine_transform(matrix, bias, inputs)

        output_noise = (numpy.zeros(outputs.shape, dtype=dtype)
                        if output_variance == 0.0
                        else rng.normal(scale=output_variance,
                                        size=outputs.shape))

        cast = numpy.cast[dtype]

        return cast(inputs), cast(outputs + output_noise)

    rng = numpy.random.RandomState(352351)

    # The "ground truth" affine transform
    matrix = rng.uniform(size=(2, 1))
    bias = rng.uniform(size=(1, ))

    min_input = (-1.0, -1.0)
    max_input = (1.0, 1.0)
    output_variance = .1

    training_inputs, training_outputs = make_random_dataset(matrix,
                                                            bias,
                                                            min_input,
                                                            max_input,
                                                            output_variance,
                                                            rng,
                                                            args.training_size,
                                                            floatX)

    testing_inputs, testing_outputs = make_random_dataset(matrix,
                                                          bias,
                                                          min_input,
                                                          max_input,
                                                          output_variance,
                                                          rng,
                                                          args.testing_size,
                                                          floatX)

    def make_grid(matrix,
                  bias,
                  min_input,
                  max_input,
                  samples_per_dimension):
        '''
        Creates a grid plane that can be given to matplotlib's plot_surface()

        Computes z values Z over a grid of X = [x, y] values , as:
        Z = numpy.dot(X, M) + B

        Parameters
        ----------
        matrix: numpy.ndarray
          M in the above equation. shape = (2, 1)
        bias: numpy.ndarray
          B in the above equation. shape = (1, )
        min_input: numpy.ndarray
          minimum X values. Shape = (2, )
        max_output: numpy.ndarray
          maximum X values. Shape = (2, )
        samples_per_dimension: int
          Number of sample points along x or y dimension of grid.
        '''
        assert_equal(matrix.shape, (2, 1))
        assert_equal(len(min_input), 2)
        assert_equal(len(max_input), 2)

        # pylint: disable=invalid-name
        xs, ys = (numpy.linspace(min_input[i],
                                 max_input[i],
                                 samples_per_dimension)
                  for i in range(2))

        # pylint: disable=unbalanced-tuple-unpacking
        grid_xs, grid_ys = numpy.meshgrid(xs, ys)

        inputs = numpy.vstack((grid_xs.flat, grid_ys.flat)).T

        grid_zs = affine_transform(matrix,
                                   bias,
                                   inputs).reshape(grid_xs.shape)

        return grid_xs, grid_ys, grid_zs

    figure = pyplot.gcf()
    figure.set_size_inches(18, 6, forward=True)

    points_axes = figure.add_subplot(1, 2, 1, projection='3d')
    points_axes.set_aspect('equal')

    logger_axes = figure.add_subplot(1, 2, 2)
    logger_axes.set_xlabel('# of epochs')
    logger_axes.set_ylabel('Average loss per sample')

    ground_truth_plane = tuple(make_grid(matrix,
                                         bias,
                                         min_input,
                                         max_input,
                                         10))

    points_axes.plot_surface(ground_truth_plane[0],
                             ground_truth_plane[1],
                             ground_truth_plane[2],
                             color=[0, 1, 0, 0.5])  # translucent

    z_ticks = (0.5 *
               numpy.arange(numpy.floor(ground_truth_plane[2].min() / 0.5),
                            numpy.ceil(ground_truth_plane[2].max() / 0.5)))
    points_axes.set_zticks(z_ticks)

    points_axes.scatter(training_inputs[:, 0],
                        training_inputs[:, 1],
                        training_outputs,
                        c='blue')

    points_axes.scatter(testing_inputs[:, 0],
                        testing_inputs[:, 1],
                        testing_outputs,
                        c='red')

    training_set, testing_set = (
        Dataset(names=('inputs', 'targets'),
                formats=(DenseFormat(axes=('b', 'f'),
                                     shape=(-1, 2),
                                     dtype=floatX),
                         DenseFormat(axes=['b', 'f'],
                                     shape=[-1, 1],
                                     dtype=floatX)),
                tensors=(inputs, outputs))
        for inputs, outputs in safe_izip((training_inputs, testing_inputs),
                                         (training_outputs, testing_outputs)))

    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = training_outputs.shape[0]

    training_iterator = training_set.iterator(iterator_type='sequential',
                                              batch_size=batch_size)
    input_node, label_node = training_iterator.make_input_nodes()

    affine_node = AffineTransform(input_node=input_node,
                                  output_format=DenseFormat(axes=['b', 'f'],
                                                            shape=[-1, 1],
                                                            dtype=None))

    loss_node = L2Loss(affine_node, label_node)
    loss_node.output_symbol.name = 'loss'

    batch_loss = loss_node.output_symbol.sum()

    grad = theano.gradient.grad

    parameter_updaters = [SgdParameterUpdater(p,
                                              grad(batch_loss, p),
                                              (args.learning_rate /
                                               training_outputs.shape[0]),
                                              args.momentum,
                                              args.nesterov)
                          for p in (affine_node.linear_node.params,
                                    affine_node.bias_node.params)]

    training_loss_logger = LogsToLists()
    validation_loss_logger = LogsToLists()

    def plot_model_surface():
        '''
        Plots the surface predicted by the model's current parameters.

        Returns
        -------
        model_surface:
          The matplotlib 3D surface object that got plotted.
        '''
        xs, ys, zs = \
            make_grid(affine_node.linear_node.params.get_value(),
                      affine_node.bias_node.params.get_value(),
                      min_input,
                      max_input,
                      10)
        model_surface = points_axes.plot_surface(xs, ys, zs,
                                                 color=[1, .5, .5, .5])
        return model_surface

    class Replotter(IterationCallback):
        '''
        Callback that replots the model surface after each batch.
        '''

        def __init__(self):
            super(Replotter, self).__init__()
            self.model_surface = [plot_model_surface()]
            self.logger_plots = []
            points_axes.set_title("Hit space to optimize. q to quit.")

        def _on_iteration(self, computed_values):
            assert_equal(len(computed_values), 0)

            if len(self.model_surface) > 0:
                self.model_surface[0].remove()
                del self.model_surface[0]

            self.model_surface.append(plot_model_surface())
            # points_axes.autoscale_view(tight=True)
            # points_axes.autoscale(enable=True, axis='both')
            # print("updated")
            points_axes.set_zlim(bottom=-1, top=1)

            # points_axes.relim()
            #points_axes.draw_idle()

            figure.canvas.draw()

        def on_epoch(self):
            for plot in self.logger_plots:
                plot.remove()

            self.logger_plots = []

            def plot_log(log, color):
                values = numpy.concatenate(log, axis=0)
                self.logger_plots.extend(logger_axes.plot(
                    numpy.arange(1, 1 + values.size),
                    values,
                    '%s-' % color))

            plot_log(training_loss_logger.log, 'b')
            plot_log(validation_loss_logger.log, 'r')

            logger_axes.legend(self.logger_plots, ["train loss", "test loss"])
            return ()

    assert_greater(batch_size, 0)

    input_symbols = [n.output_symbol for n in (input_node, label_node)]

    training_loss_monitor = MeanOverEpoch(loss_node,
                                          callbacks=[training_loss_logger])

    training_stopper = StopsOnStagnation(max_epochs=10,
                                         min_proportional_decrease=.01)
    validation_loss_monitor = MeanOverEpoch(loss_node,
                                            callbacks=[validation_loss_logger,
                                                       training_stopper])

    validation_callback = ValidationCallback(
        inputs=input_symbols,
        input_iterator=testing_set.iterator(
            iterator_type='sequential',
            batch_size=testing_inputs.shape[0]),
        epoch_callbacks=[validation_loss_monitor])

    sgd = Sgd(inputs=[input_node, label_node],
              input_iterator=training_iterator,
              callbacks=(parameter_updaters + [training_loss_monitor,
                                               Replotter(),
                                               LimitsNumEpochs(100),
                                               validation_callback]))

    def on_key_press(event):
        '''
        Key press callback.
        '''
        if event.key == 'q':
            sys.exit(0)
        if event.key == ' ':
            sgd.train()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.grid()
    pyplot.show()


if __name__ == '__main__':
    main()
