# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.layers import recurrent
import numpy as np
from six.moves import range

from data.number_data_engine import NumberDataEngine
from data.character_data_engine import CharacterDataEngine

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.SimpleRNN
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

print('Generating data...')
engine = NumberDataEngine()
questions, expected = engine.get_dataset(TRAINING_SIZE)
print('Total addition questions:', len(questions))

print('Vectorization...')
convertor = CharacterDataEngine(engine.get_character_set(), maxlen=MAXLEN)
D_X = convertor.encode_dataset(questions, invert=True)
D_y = convertor.encode_dataset(expected, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(D_y))
np.random.shuffle(indices)
D_X = D_X[indices]
D_y = D_y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(D_X) - len(D_X) / 10
(D_X_train, D_X_val) = (slice_X(D_X, 0, split_at), slice_X(D_X, split_at))
(D_y_train, D_y_val) = (D_y[:split_at], D_y[split_at:])

print(D_X_train.shape)
print(D_y_train.shape)





import lx_layer.layer as L
import lx_layer.recurrent as R
import lx_layer.attention as A
import theano.printing as P
from keras import activations, objectives
from keras import models
from keras.optimizers import Adam
from keras import backend as K
from util import initializations

print('Build model...')
input_dim = convertor.get_dim()
output_dim = convertor.get_dim()
hidden_dim = HIDDEN_SIZE
attention_hidden_dim = 15
input_length = MAXLEN
output_length = DIGITS + 1

'''Define functions'''
activation_softmax = activations.get('softmax')
activation_linear = activations.get('linear')
activation_hard_sigmoid = activations.get('hard_sigmoid')
activation_tanh = activations.get('tanh')
init_glorot_uniform = initializations.glorot_normal
init_orthogonal = initializations.orthogonal
forget_bias_init= initializations.one

'''Define layers'''
#Data layer
data = L.Input(3, 'X')
#RNN encoder
encoder = R.LSTM(input_length, input_dim, hidden_dim, name='ENCODER')
encoder.set_function('activation', activation_tanh)
encoder.set_function('inner_activation', activation_hard_sigmoid)
encoder.set_function('init', init_glorot_uniform)
encoder.set_function('inner_init', init_orthogonal)
encoder.set_function('forget_bias_init', forget_bias_init)
#Repeat Vector
repeater = L.RepeatVector(output_length)
#RNN decoder
decoder = A.AttentionLSTM(output_length, hidden_dim, hidden_dim, input_dim, attention_hidden_dim, name='ATT')
decoder.set_function('activation', activation_tanh)
decoder.set_function('attention_activation', activation_tanh)
decoder.set_function('inner_activation', activation_hard_sigmoid)
decoder.set_function('init', init_glorot_uniform)
decoder.set_function('inner_init', init_orthogonal)
decoder.set_function('forget_bias_init', forget_bias_init)
#Time Distributed Dense
tdd = L.TimeDistributedDense(output_length, hidden_dim, output_dim, 'TDD')
tdd.set_function('activation', activation_linear)
tdd.set_function('init', init_glorot_uniform)
#Activation
activation = L.Activation()
activation.set_function('activation', activation_softmax)
#Output layer
output = L.Output()

'''Define Relations'''
encoder.set_input('input_sequence', data, 'output')
repeater.set_input('input', encoder, 'output_last')
decoder.set_input('context', data, 'output')
decoder.set_input('input_sequence', repeater, 'output')
tdd.set_input('input', decoder, 'output_sequence')
activation.set_input('input', tdd, 'output')
output.set_input('input', activation, 'output')

layers = [data, encoder, repeater, decoder, tdd, activation, output]

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
print('y_train:')
print(P.debugprint(y_train))
print('y_test:')
print(P.debugprint(y_test))


'''loss'''

loss = objectives.get('categorical_crossentropy')
weighted_loss = models.weighted_objective(loss)
y = K.placeholder(ndim=K.ndim(y_train))
weights = K.placeholder(ndim=1)
train_loss = weighted_loss(y, y_train, weights, mask_train)
test_loss = weighted_loss(y, y_test, weights, mask_test)

_y_train = K.placeholder(ndim=3, name='y_train')
_y_test = K.placeholder(ndim=3, name='y_test')
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
optimizer = Adam()
_updates = optimizer.get_updates(params, constraints, train_loss)
updates += _updates

print('after Adam, updates:')
for update in updates:
    print(update)

train_ins = [X_train, y, weights]
test_ins = [X_test, y, weights]
predict_ins = [X_test]

'''Get functions'''
print('complie: _train')
_train = K.function(train_ins, [train_loss], updates=updates)
print('complie: _train_with_acc')
_train_with_acc = K.function(train_ins, [train_loss, train_accuracy], updates=updates)
print('complie: _predict')
_predict = K.function(predict_ins, [y_test], updates=state_updates)
print('complie: _test')
_test = K.function(test_ins, [test_loss])
print('complie: _test_with_acc')
_test_with_acc = K.function(test_ins, [test_loss, test_accuracy])

model = Sequential()
model.class_mode = "categorical"
model._train = _train
model._train_with_acc = _train_with_acc
model._predict = _predict
model._test = _test
model._test_with_acc = _test_with_acc













# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(D_X_train, D_y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(D_X_val, D_y_val), show_accuracy=True)
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(D_X_val))
        rowX, rowy = D_X_val[np.array([ind])], D_y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = convertor.decode(rowX[0], invert=True)
        correct = convertor.decode(rowy[0])
        guess = convertor.decode(preds[0], calc_argmax=False)
        print('Q', q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
