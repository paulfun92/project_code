import numpy

import theano
import theano.tensor as tt
import theano.gradient

input_size = 4
hidden_size = 3
output_size = 2
Wh_flat = theano.shared(numpy.random.randn(input_size * hidden_size).astype(theano.config.floatX))
bh = theano.shared(numpy.zeros(hidden_size, dtype=theano.config.floatX))
Wy_flat = theano.shared(numpy.random.randn(hidden_size * output_size).astype(theano.config.floatX))
by = theano.shared(numpy.zeros(output_size, dtype=theano.config.floatX))
parameters = [Wh_flat, bh, Wy_flat, by]
Wh = Wh_flat.reshape((input_size, hidden_size))
Wy = Wy_flat.reshape((hidden_size, output_size))
x = tt.matrix(dtype=theano.config.floatX)
z = tt.matrix(dtype=theano.config.floatX)
h = tt.nnet.sigmoid(theano.dot(x, Wh) + bh)
y = tt.exp(theano.dot(h, Wy) + by)/tt.exp(theano.dot(h, Wy) + by).sum(axis=1, keepdims=True)
c = tt.nnet.categorical_crossentropy(y, z).mean()
gs = theano.grad(c, parameters)
hs = theano.gradient.hessian(c, parameters)
f = theano.function([x, z], [y, c] + gs + hs)
batch_size = 5
print f(numpy.random.randn(batch_size, input_size).astype(theano.config.floatX),
        numpy.random.randn(batch_size, output_size).astype(theano.config.floatX))