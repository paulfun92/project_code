import h5py
import numpy
import sys
import os
from matplotlib import pyplot
from nose.tools import assert_equal


directory = '/media/paul/DRIVE-N-GO/Backup Project NN/NEW OUTPUT_/ALL_OUTPUT_NEW/MNIST/ConvNets/SGD_methods'
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
    #name = name[3:]
    name = name.replace('_',' + ')

    data_plot.append(data)
    cumm_times.append(cumm_time)
    names.append(name)
                      
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
axes.legend(names)
pyplot.ylabel("Validation misclassification rate")
pyplot.xlabel(xlabel_text)
pyplot.title("MNIST ConvNet: SGD methods")
pyplot.show()
