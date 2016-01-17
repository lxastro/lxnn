from theano import tensor as T 
from theano import config
import theano.printing as P

dtype = config.floatX

def theano_variable(ndim, name=None, dtype=dtype):
    '''Instantiate an theano variable.
    '''
    if ndim == 0:
        return T.scalar(name=name, dtype=dtype)
    elif ndim == 1:
        return T.vector(name=name, dtype=dtype)
    elif ndim == 2:
        return T.matrix(name=name, dtype=dtype)
    elif ndim == 3:
        return T.tensor3(name=name, dtype=dtype)
    elif ndim == 4:
        return T.tensor4(name=name, dtype=dtype)
    else:
        raise Exception('ndim too large: ' + str(ndim))
    
def repeat(x, n):
    '''
    # Input shape
        2D tensor of shape `(d1, d2)`.
    # Output shape
        3D tensor of shape `(n, d1, d2)`.
    '''
    tensors = [x] * n
    stacked = T.stack(*tensors)
    return stacked
    
def _test_theano_variable():
    print '\n------------------------------------------------------------'
    print 'Test: theano_variable'
    x0 = theano_variable(0, 'x0')
    x1 = theano_variable(1, 'x1')
    x2 = theano_variable(2, 'x2')
    x3 = theano_variable(3, 'x3')
    x4 = theano_variable(4, 'x4')
    print x0, type(x0)
    print x1, type(x1)
    print x2, type(x2)
    print x3, type(x3)
    print x4, type(x4)

def _test_repeat():
    print '\n------------------------------------------------------------'
    print 'Test: theano_variable'
    x2 = theano_variable(2, 'x2')
    x3 = theano_variable(3, 'x3')
    y3 = repeat(x2, 10)
    y4 = repeat(x3, 10)
    print P.pprint(y3)
    print P.pprint(y4)

if __name__ == '__main__':
    _test_theano_variable()
    _test_repeat()