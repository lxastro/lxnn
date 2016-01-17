'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils



nb_classes = 10
# the data, shuffled and split between tran and test sets
(D_X_train, D_y_train), (D_X_test, D_y_test) = mnist.load_data()

D_X_train = D_X_train.reshape(60000, 784)
D_X_test = D_X_test.reshape(10000, 784)
D_X_train = D_X_train.astype('float32')
D_X_test = D_X_test.astype('float32')
D_X_train /= 255
D_X_test /= 255
print(D_X_train.shape[0], 'train samples')
print(D_X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
D_Y_train = np_utils.to_categorical(D_y_train, nb_classes)
D_Y_test = np_utils.to_categorical(D_y_test, nb_classes)



import lx_layer.layer as L
import theano.printing as P
from keras import activations, objectives
from keras import models
from keras.optimizers import RMSprop
from keras import backend as K
from util import initializations


input_size = 784
hidden_size = 512
output_size = nb_classes
batch_size = 128
nb_epoch = 20


'''Define functions'''
mask_none = lambda :None
activation_linear = activations.get('linear')
activation_relu = activations.get('relu')
activation_softmax = activations.get('softmax')
init_glorot_uniform = initializations.glorot_uniform

'''Define layers'''
#Data layer
data = L.Input(2, 'X')
data.set_function('mask', mask_none)
#Dense layer 1
dense_1 = L.Dense(input_size, hidden_size, 'Dense1')
dense_1.set_function('activation', activation_linear)
dense_1.set_function('init', init_glorot_uniform)
#Dense layer 2
dense_2 = L.Dense(hidden_size, hidden_size, 'Dense2')
dense_2.set_function('activation', activation_linear)
dense_2.set_function('init', init_glorot_uniform)
#Dense layer 3
dense_3 = L.Dense(hidden_size, output_size, 'Dense3')
dense_3.set_function('activation', activation_linear)
dense_3.set_function('init', init_glorot_uniform)
#Activation layer 1
activation_1 = L.Activation()
activation_1.set_function('activation', activation_relu)
#Activation layer 2
activation_2 = L.Activation()
activation_2.set_function('activation', activation_relu)
#Activation layer 3
activation_3 = L.Activation()
activation_3.set_function('activation', activation_softmax)
#Output layer
output = L.Output()

'''Define Relations'''
dense_1.set_input('input', data, 'output')
activation_1.set_input('input', dense_1, 'output')
dense_2.set_input('input', activation_1, 'output')
activation_2.set_input('input', dense_2, 'output')
dense_3.set_input('input', activation_2, 'output')
activation_3.set_input('input', dense_3, 'output')
output.set_input('input', activation_3, 'output')

layers = [data, dense_1, dense_2, dense_3, activation_1, activation_2, activation_3, output]

'''Build Layers'''
for layer in layers:
    layer.build()

'''input, output'''
# input of model
X_train = data.input(train=True)
X_test = data.input(train=False)
# output of model
y_train = output.output(train=True)
y_test = output.output(train=False)
mask_train = output.output_mask(train=True) # None in this example
mask_test = output.output_mask(train=False) # None in this example

print('X_train:', P.pprint(X_train))
print('X_test:', P.pprint(X_test))
print('y_train:', P.pprint(y_train))
print('y_test:', P.pprint(y_test))

'''loss'''

loss = objectives.get('categorical_crossentropy')
weighted_loss = models.weighted_objective(loss)
y = K.placeholder(ndim=K.ndim(y_train))
weights = K.placeholder(ndim=1)
train_loss = weighted_loss(y, y_train, weights, mask_train)
test_loss = weighted_loss(y, y_test, weights, mask_test)

_y_train = K.placeholder(ndim=2, name='y_train')
_y_test = K.placeholder(ndim=2, name='y_test')
_train_loss = weighted_loss(y, _y_train, weights, mask_train)
_test_loss = weighted_loss(y, _y_test, weights, mask_test)
print('train_loss:', P.pprint(_train_loss))
print('test_loss', P.pprint(_test_loss))

'''categorical accuracy'''
train_accuracy = K.mean(K.equal(K.argmax(y, axis=-1), K.argmax(y_train, axis=-1)))
test_accuracy = K.mean(K.equal(K.argmax(y, axis=-1), K.argmax(y_test, axis=-1)))

_train_accuracy = K.mean(K.equal(K.argmax(y, axis=-1), K.argmax(_y_train, axis=-1)))
_test_accuracy = K.mean(K.equal(K.argmax(y, axis=-1), K.argmax(_y_test, axis=-1)))
print('train_accuracy:', P.pprint(_train_accuracy))
print('test_accuracy', P.pprint(_test_accuracy))

'''parameters'''
params = []
regularizers = []
constraints = []
updates = []
state_updates = []
for layer in layers:
    _params, _regularizers, _consts, _updates = layer.get_params()
    params += _params
    regularizers += _regularizers
    constraints += _consts
    updates += _updates
    
print('parameters:')
print(params)
print('regularizers:')
print(regularizers)
print('constrains:')
print(constraints)
print('updates:')
print(updates)

'''updates'''
optimizer = RMSprop()
_updates = optimizer.get_updates(params, constraints, train_loss)
updates += _updates

print('after RMSprop, updates:')
for update in updates:
    print(update)

train_ins = [X_train, y, weights]
test_ins = [X_test, y, weights]
predict_ins = [X_test]

'''Get functions'''
_train = K.function(train_ins, [train_loss], updates=updates)
_train_with_acc = K.function(train_ins, [train_loss, train_accuracy], updates=updates)
_predict = K.function(predict_ins, [y_test], updates=state_updates)
_test = K.function(test_ins, [test_loss])
_test_with_acc = K.function(test_ins, [test_loss, test_accuracy])


model = Sequential()
model._train = _train
model._train_with_acc = _train_with_acc
model._predict = _predict
model._test = _test
model._test_with_acc = _test_with_acc


model.fit(D_X_train, D_Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(D_X_test, D_Y_test))
score = model.evaluate(D_X_test, D_Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

