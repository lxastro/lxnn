import layer as L
import recurrent as R
import theano.printing as P
from keras import activations
from util import initializations
from util import theano_util as UL
import theano

activation_sigmoid = activations.get('sigmoid')
init_glorot_uniform = initializations.glorot_normal
init_orthogonal = initializations.orthogonal

def _test_RNN1():
    print '\n------------------------------------------------------------'
    print 'Test: RNN layer 1'   
    X = L.Input(3, name='DATA3', mask_dim=2)
    rnn1 = R.RNN(3,1024,10, name='RNN1')
    rnn1.set_function('activation', activation_sigmoid)
    rnn1.set_function('init', init_glorot_uniform)
    rnn1.set_function('inner_init', init_orthogonal)
    rnn1.set_input('input_sequence', X, 'output')
    X.build()
    rnn1.build()
    print 'Test mask:', X.input_mask(train=False)
    print 'Test output_last:'
    print P.debugprint(rnn1.get_output('output_last', train=False))
    print 'Test output_sequence:'
    print P.debugprint(rnn1.get_output('output_sequence', train=False))

    print 'Train mask:', P.pprint(X.input_mask(train=True))
    print 'Train output_last:'
    print P.debugprint(rnn1.get_output('output_last', train=True))
    print 'Train output_sequence:'
    print P.debugprint(rnn1.get_output('output_sequence', train=True))

def _test_RNN2(): 
    print '\n------------------------------------------------------------'
    print 'Test: RNN layer 2'  
    D = L.Input(2, name='DATA2')
    rnn2 = R.RNN(10,1024,10, name='RNN2')
    rnn2.set_function('activation', activation_sigmoid)
    rnn2.set_function('init', init_glorot_uniform)
    rnn2.set_function('inner_init', init_orthogonal)
    rnn2.set_input('input_single', D, 'output')
    D.build()
    rnn2.build()
    print 'Test mask 2:', D.input_mask(train=False)
    print 'Test output_last 2:'
    print P.debugprint(rnn2.get_output('output_last', train=False))
    print 'Test output_sequence 2:'
    print P.debugprint(rnn2.get_output('output_sequence', train=False))

    print 'Train mask 2:', D.input_mask(train=True)
    print 'Train output_last 2:'
    print P.debugprint(rnn2.get_output('output_last', train=True))
    print 'Train output_sequence 2:'
    print P.debugprint(rnn2.get_output('output_sequence', train=True))

    
if __name__ == '__main__':
    _test_RNN1()
    _test_RNN2()