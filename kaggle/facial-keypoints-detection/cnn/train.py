import sys
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

import dataset

ds = dataset.get_data()

learning_rate = 1e-1

n_outputs = 30

identity = lambda x: x

rng = np.random.RandomState(12345)

parameters = []

def conv2d(name, x, input_channel, output_channel, kernel_size, padding=False, nl=identity, W=None, b=None):
    def GaussianParamInitializer(rng, shape, mean=0, std=0.03):
        param = np.asarray(
            rng.normal(loc=mean, scale=std, size=shape), 
            dtype=theano.config.floatX
        )
        return param

    W_value = GaussianParamInitializer(rng, shape=(output_channel, input_channel, kernel_size, kernel_size))\
                  if W is None else W
    b_value = GaussianParamInitializer(rng, std=0.001, shape=(output_channel,)) if b is None else b

    W = theano.shared(value=W_value, name=name+':W', borrow=True)
    b = theano.shared(value=b_value, name=name+':b', borrow=True)
    b_4d = b[None, :, None, None]

    parameters.extend([W, b])

    return nl(T.nnet.conv2d(x, W, border_mode='half' if padding else 'valid') + b_4d)

def max_pooling(name, x, window, stride=None):
    if isinstance(window, int):
        window = (window, window)
    if isinstance(stride, int):
        stride = (stride, stride)
    if stride is None:
        stride = window
    from theano.tensor.signal.pool import pool_2d
    return pool_2d(x, window, ignore_border=False, st=stride)

def fully_connected(name, x, inputs, outputs, W=None, b=None, nl=identity):
    def W_initializer(rng, n_in, n_out):
        param = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)), 
                high=np.sqrt(6. / (n_in + n_out)), 
                size=(n_in, n_out), 
            ), 
            dtype=theano.config.floatX, 
        )
        return param

    W_value = W_initializer(rng, inputs, outputs) if W is None else W
    b_value = np.zeros(outputs) if b is None else b

    W = theano.shared(value=W_value, name=name+':W', borrow=True)
    b = theano.shared(value=b_value, name=name+':b', borrow=True)
    
    parameters.extend([W, b])

    return nl(T.dot(x, W) + b)

# training configurations
nepochs = 10
batch_size = 32

data = T.matrix("data")
label = T.matrix("label")
mask = T.matrix("mask")

def logsoftmax(x):
    x -= x.max(axis=1, keepdims=True)
    return x - T.log(T.exp(x).sum(axis=1, keepdims=True))

# construct model
data_ = data / 255.0
label_ = label / 96.0

x = data_.reshape((-1, 1, 96, 96))
x = conv2d('conv1_1', x, 1, 32, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv1_2', x, 32, 32, kernel_size=3, padding=True, nl=T.nnet.relu)
x = max_pooling('pool1', x, 2)
x = conv2d('conv2_1', x, 32, 64, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv2_2', x, 64, 64, kernel_size=3, padding=True, nl=T.nnet.relu)
x = max_pooling('pool2', x, 2)
x = conv2d('conv3_1', x, 64, 128, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv3_2', x, 128, 128, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv3_3', x, 128, 128, kernel_size=3, padding=True, nl=T.nnet.relu)
x = max_pooling('pool3', x, 2)
x = conv2d('conv4_1', x, 128, 256, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv4_2', x, 256, 256, kernel_size=3, padding=True, nl=T.nnet.relu)
x = conv2d('conv4_3', x, 256, 256, kernel_size=3, padding=True, nl=T.nnet.relu)
x = max_pooling('pool4', x, 2)
x = x.flatten(ndim=2)
x = fully_connected('fc1', x, 256*6*6, 1024, nl=T.nnet.relu)
x = fully_connected('fc2', x, 1024, n_outputs)

l2_loss = ((x * mask - label_) ** 2).mean()

gradients = OrderedDict(zip(parameters, T.grad(l2_loss, parameters)))
steps = [(parameter, parameter - learning_rate * gradient)
         for parameter, gradient in gradients.items()]

# compile theano functions
monitor_fn = theano.function([data, label, mask], l2_loss)
step_fn = theano.function([data, label, mask], updates=steps)

# training process
def slice_sources(dataset, *slice_args):
    s = slice(*slice_args)
    return dict((source_name, source[s])
                for source_name, source in dataset.items())

for i in range(nepochs):
    # print(i, "train cross entropy", monitor_fn(**ds["train"]))
    print(i, "training")
    total_minibatch = len(ds["train"]["data"])
    for j, a in enumerate(range(0, total_minibatch, batch_size)):
        b = a + batch_size
        step_fn(**slice_sources(ds["train"], a, b))
        print(a, "/", total_minibatch, end='\r')
        sys.stdout.flush()
    print()
    print(i, "done")
