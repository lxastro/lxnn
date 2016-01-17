from data.simple_chain_engine import SimpleChainEngine
from data.character_data_engine import CharacterDataEngine
from keras import backend as K
from keras.layers.core import Activation, TimeDistributedDense
from keras_layer.shift import Shift
from keras.layers.recurrent import SimpleRNN
import numpy as np
from keras.layers.containers import Graph

TRAINING_SIZE = 100
chars = '0123456789abcdef'

print('Generating data...')
engine = SimpleChainEngine(chars)
starts, chains = engine.get_dataset(TRAINING_SIZE)
print('Total number of data:', len(starts))

print('Vectorization...')
convertor = CharacterDataEngine(chars, maxlen=len(chars)-1)
initial_value = convertor.encode_dataset(starts, maxlen=1)
y = convertor.encode_dataset(chains)
split_at = len(y) - len(y) / 10
(y_train, y_val) = (y[:split_at], y[split_at:])
(i_train, i_val) = (initial_value[:split_at], initial_value[split_at:])
(X_train, X_val) = (y_train, y_val)
print(i_train.shape)
print(y_train.shape)

print('Build model...')
HIDDEN_SIZE = 128
BATCH_SIZE = 50
MAXLEN = len(chars)-1
input_dim = convertor.get_dim()
rnn_layer = SimpleRNN(HIDDEN_SIZE, input_shape=(MAXLEN, convertor.get_dim()), return_sequences=True)
shift_layer = Shift(rnn_layer, initial_value)
model = Graph()
model.add_input(name='initial_value', input_shape=(1,input_dim))
model.add_input(name='sequence_input', input_shape=(MAXLEN,input_dim))
model.add_node(shift_layer, name='shift', input='sequence_input')
model.add_node(rnn_layer, name = 'rnn', input='shift')
model.add_node(TimeDistributedDense(input_dim), name = 'tdd', input = 'rnn')
model.add_node(Activation('softmax'), name = 'softmax', input = 'tdd')
model.add_output(name='output', input='softmax')

model.compile(loss = {'output':'categorical_crossentropy'}, optimizer='adam')

for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit({'initial_value':i_train, 'sequence_input':X_train, 'output':y_train}, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data={'initial_value':i_val, "sequence_input":X_val}, show_accuracy=True)

    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        rowi, rowX, rowy = i_val[np.array([i])], X_val[np.array([i])], y_val[np.array([i])]
        proba = model.predict_classes({'initial_value':rowi, "sequence_input":rowX}, verbose=0)['output']
        preds = proba.argmax(axis=-1)
        start = convertor.decode(rowX[0])
        correct = convertor.decode(rowy[0])
        guess = convertor.decode(preds[0], calc_argmax=False)
        
        print('Start  : ', start)
        print('Correct: ', correct)
        print('Gusee  : ', guess)
        print('---')
        
        