import layer as L
import theano.printing as P
from keras import activations
from util import initializations
from util import theano_util as UL

activation_linear = activations.get('linear')
activation_relu = activations.get('relu')
activation_softmax = activations.get('softmax')
init_glorot_uniform = initializations.glorot_normal

def _test_function():
    print '\n------------------------------------------------------------'
    print 'Test: activation, initialization functions'
    X = UL.theano_variable(2, 'X')
    linear = activation_linear(X)
    relu = activation_relu(X)
    softmax = activation_softmax(X)
    print P.pprint(linear)
    print P.pprint(relu)
    print P.pprint(softmax)
       
    W = init_glorot_uniform((16, 24), 'W')
    print(P.pprint(W))

def _test_Input():
    print '\n------------------------------------------------------------'
    print 'Test: Input layer'
    data1 = L.Input(2, name='Data1')
    data1.build()
    print P.pprint(data1.input())
    data2 = L.Input(2)
    data2.build()
    print P.pprint(data2.input())

    X = L.Input(3, name='X', mask_dim=2)
    X.build()
    print P.pprint(X.input())
    print P.pprint(X.input_mask(train=True))
    print X.input_mask(train=False)
   
def _test_Dense():
    print '\n------------------------------------------------------------'
    print 'Test: Dense layer'
    data_1 = L.Input(2, name='Data1')
    dense_1 = L.Dense(16,24, name='Dense1')
    dense_1.set_function('activation', activation_linear)
    dense_1.set_function('init', init_glorot_uniform)
    dense_1.set_input('input', data_1, 'output')
    data_1.build()
    dense_1.build()
    print P.pprint(dense_1.get_output('output'))

def _test_Dropout():
    print '\n------------------------------------------------------------'
    print 'Test: Dropout layer'
    data_1 = L.Input(2, name='X')
    dropout = L.Dropout(0.2)
    dropout.set_input('input', data_1, 'output')
    data_1.build()
    dropout.build()
    print P.pprint(dropout.get_output('output', train=False))  
    print P.pprint(dropout.get_output('output', train=True))
    
def _test_Activation():
    print '\n------------------------------------------------------------' 
    print 'Test: Activation Layer'
    x = L.Input(2, name='X')
    relu = L.Activation()
    relu.set_function('activation', activation_relu)
    relu.set_input('input', x, 'output')
    softmax = L.Activation()
    softmax.set_function('activation', activation_softmax)
    softmax.set_input('input', x, 'output')
    x.build()
    relu.build()
    softmax.build()
    print P.pprint(relu.get_output('output'))
    print P.pprint(softmax.get_output('output'))

def _test_Output():
    print '\n------------------------------------------------------------' 
    print 'Test: Output Layer'
    x = L.Input(2, name='X')
    output = L.Output()
    output.set_input('input', x, 'output')
    x.build()
    output.build()
    print P.pprint(output.output())
    
    
def _test_Lambda():
    print '\n------------------------------------------------------------' 
    print 'Test: Lambda Layer'
    x = L.Input(2, name='X')
    y = L.Input(2, name='Y')
    def fun(x, y):
        return x*2, x+y, y*2
    f = L.Lambda(fun, ['2x', 'x+y', '2y'])
    f.set_input('input_x', x, 'output')
    f.set_input('input_y', y, 'output')
    x.build()
    y.build()
    f.build()
    print P.pprint(f.get_output('2x'))
    print P.pprint(f.get_output('x+y'))
    print P.pprint(f.get_output('2y'))

    output1 = L.Output()
    output1.set_input('input', f, '2x')
    output1.build()
    print P.pprint(output1.output())
    
def _test_SimpleLambda():
    print '\n------------------------------------------------------------' 
    print 'Test: Simple Lambda Layer'
    x = L.Input(2, name='X')
    def fun(x):
        return x**2
    f = L.SimpleLambda(fun)
    f.set_input('input', x, 'output')
    x.build()
    f.build()
    print P.pprint(f.get_output('output'))
    
def _test_RepeatVector():
    print '\n------------------------------------------------------------' 
    print 'Test: Repeat Vector Layer'
    x = L.Input(2, name='X')
    f = L.RepeatVector(10)
    f.set_input('input', x, 'output')
    x.build()
    f.build()
    print P.pprint(f.get_output('output'))

def _test_TimeDistributedDense():
    print '\n------------------------------------------------------------' 
    print 'Test: Time Distributed Dense Layer'
    x = L.Input(3, name='X')
    tdd = L.TimeDistributedDense(16,1024,128)
    tdd.set_function('activation', activation_linear)
    tdd.set_function('init', init_glorot_uniform)
    tdd.set_input('input', x, 'output')
    x.build()
    tdd.build()
    print P.debugprint(tdd.get_output('output'))

if __name__ == '__main__':
    _test_function()
    _test_Input()
    _test_Dense()
    _test_Dropout()
    _test_Activation()
    _test_Output()
    _test_Lambda()
    _test_SimpleLambda()
    _test_RepeatVector()
    _test_TimeDistributedDense()