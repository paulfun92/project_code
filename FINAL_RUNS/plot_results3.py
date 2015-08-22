import h5py
import numpy
import sys
import os
from matplotlib import pyplot
from nose.tools import assert_equal


#directory = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/CIFAR-10/ConvNets/quasi_newton_methods'
directory = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/CIFAR-10/ConvNets/quasi_newton_methods'
benchmark = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/CIFAR-10/ConvNets/SGD_methods/SGD.h5'
#benchmark2 = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/MNIST/ConvNets/quasi_newton_methods/GD.h5'
#directory = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/MNIST/ConvNets/SGD_methods'
#benchmark = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/MNIST/ConvNets/quasi_newton_methods/GD.h5'
time_minutes = True

if time_minutes:
    xlabel_text = "Time (minutes)"
else:
    xlabel_text = "Time (seconds)"

title_string = directory[-35:]

while "/" in title_string:
    title_string = title_string[1:]

title_string = title_string.replace('_', ' ')

inputs = os.listdir(directory)
print(inputs)

logs = []
data_plot = []
cumm_times = []
names = []

for i in range(len(inputs)):
    input_ = directory + '/' + inputs[i]
    h5_file = h5py.File(input_, mode='r')
    logs = h5_file['logs']
    data = numpy.asarray(logs['validation misclassification'])
    time = numpy.asarray(logs['epoch duration'])
    cumm_time = numpy.insert(numpy.cumsum(time), 0, 0)
    
    if data.shape[0] != cumm_time.shape[0]:
        data = numpy.delete(data, -1)

    if data.shape[0] != cumm_time.shape[0]:
        raise Exception('Array lengths of data and cummulative time are still not equal. Check again your results.')

    if time_minutes:
        cumm_time = cumm_time / 60

    name = inputs[i]
    name = name[:-3]
    name = name.replace('_plus', ' plus')
    #name = name.replace('_',' + ')

    if name == 'L-BFGS':
        until = 95
        data = data[:until]
        cumm_time = cumm_time[:until]

    data_plot.append(data)
    cumm_times.append(cumm_time)
    names.append(name)

h5_file = h5py.File(benchmark, mode='r')
logs = h5_file['logs']
data = numpy.asarray(logs['validation misclassification'])
time = numpy.asarray(logs['epoch duration'])
cumm_time = numpy.insert(numpy.cumsum(time), 0, 0)

if data.shape[0] != cumm_time.shape[0]:
    data = numpy.delete(data, -1)

if data.shape[0] != cumm_time.shape[0]:
    raise Exception('Array lengths of data and cummulative time are still not equal. Check again your results.')

if time_minutes:
    cumm_time = cumm_time / 60

name = "SGD"

data_plot.append(data)
cumm_times.append(cumm_time)
names.append(name)
print(len(data_plot))

'''
h5_file = h5py.File(benchmark2, mode='r')
logs = h5_file['logs']
data = numpy.asarray(logs['validation misclassification'])
time = numpy.asarray(logs['epoch duration'])
cumm_time = numpy.insert(numpy.cumsum(time), 0, 0)

if data.shape[0] != cumm_time.shape[0]:
    data = numpy.delete(data, -1)

if data.shape[0] != cumm_time.shape[0]:
    raise Exception('Array lengths of data and cummulative time are still not equal. Check again your results.')

if time_minutes:
    cumm_time = cumm_time / 60

name = "GD"

data_plot.append(data)
cumm_times.append(cumm_time)
names.append(name)
'''

                      
assert_equal(len(data_plot), len(cumm_times))

# PLOTTING:
pyplot.hold(True)
'''
for data, time in zip(data_plot, cumm_times):
    pyplot.plot(time, data)
'''

string = ['ro-','b*-','g+-','cx-','k^-','yv-','ms-', 'ro-','b*-','g+-','cx-','k^-','yv-','ms-',  'ro-','b*-','g+-','cx-','k^-','yv-','ms-']

for i in range(len(data_plot)):
    pyplot.plot(cumm_times[i], data_plot[i], string[i], linewidth = 1.5, markevery=5)

axes = pyplot.gca()
axes.set_yscale('log')
axes.legend(names, loc=4)
pyplot.ylabel("Validation misclassification rate")
pyplot.xlabel(xlabel_text)
pyplot.title("CIFAR-10 ConvNet: L-BFGS")
pyplot.show()
