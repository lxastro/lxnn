'''
@author: Xiang Long
'''
from keras import backend as K
from keras import constraints
import theano
import util.theano_util as TU

class Layer(object):
    '''
    Abstract Layer
    '''
    def __init__(self):
        '''
        Define inputs_dict, outputs_dict, functions_dict
        '''
        self.inputs_dict = {}
        self.outputs_dict = {}
        self.functions_dict = {}
        self.input_names = []
        self.output_names = []
        self.function_names = []

    def set_input(self, input_name, input_layer, output_name):
        self.inputs_dict[input_name] = (input_layer, output_name)
        
    def set_output(self, output_name, output_function, output_mask_function):
        self.outputs_dict[output_name] = (output_function, output_mask_function) 
    
    def set_function(self, function_name, function):
        self.functions_dict[function_name] = function
    
    def get_input(self, input_name, train=False):
        (input_layer, output_name) = self.inputs_dict[input_name]
        return input_layer.get_output(output_name, train)
    
    def get_input_mask(self, input_name, train=False):
        (input_layer, output_name) = self.inputs_dict[input_name]
        return input_layer.get_output_mask(output_name, train)
    
    def get_output(self, output_name, train=False):
        return self.outputs_dict[output_name][0](train)
    
    def get_output_mask(self, output_name, train=False):
        return self.outputs_dict[output_name][1](train)
    
    def get_function(self, function_name):
        return self.functions_dict[function_name]
    
    def get_input_names(self):
        return self.input_names

    def get_output_names(self):
        return self.output_names
    
    def get_function_names(self):
        return self.function_names
    
    def _default_input(self, train=False):
        return self.get_input('input', train)
    
    def _default_input_mask(self, train=False):
        return self.get_input_mask('input', train)
    
    def _default_output_mask(self, train=False):
        return self._default_input_mask(train)
    
    def _no_output_mask(self, train=False):
        return None
    
    def build(self):
        pass
    
    def get_params(self):
        consts = []
        updates = []
        
        if hasattr(self, 'params'):
            params = self.params
        else:
            params = []

        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        if hasattr(self, 'constraints') and len(self.constraints) == len(params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif hasattr(self, 'constraint') and self.constraint:
            consts += [self.constraint for _ in range(len(params))]
        else:
            consts += [constraints.identity() for _ in range(len(params))]

        if hasattr(self, 'updates') and self.updates:
            updates += self.updates

        return params, regularizers, consts, updates
    
class Input(Layer):
    '''A layer without input for input data.
    # Output shape
        input_dim dimension tensor.
    # Input mask
        default None
    '''
    def __init__(self, input_dim, name='Input', mask_dim=None):
        super(Input, self).__init__()
        self.input_dim = input_dim
        self.name = name
        self.mask_dim=mask_dim
        
        self.input_names += []
        self.output_names += ['output']
        self.function_names += ['mask']
        self.set_output('output', self.output, self.output_mask)
        if self.mask_dim:
            self.set_function('mask', self._get_mask)
        else:
            self.set_function('mask', self._no_output_mask)
    
    def build(self):
        self.X = TU.theano_variable(self.input_dim, name=self.name)
        if self.mask_dim:
            self.m = TU.theano_variable(self.mask_dim, name='mask_'+self.name, dtype='int8')
        
    def input(self, train=False):
        return self.output(train)
    
    def input_mask(self, train=False):
        return self.output_mask(train)

    def output(self, train=False):
        return self.X
    
    def _get_mask(self, train=False):
        return self.m
        
    def output_mask(self, train=False):
        return self.get_function('mask')(train)

class Dense(Layer):
    '''Fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, input_dim, output_dim, name='Dense'):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        
        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += ['init', 'activation']
        self.set_output('output', self.output, self._no_output_mask)
        

    def build(self):
        self.W = self.get_function('init')((self.input_dim, self.output_dim), name=self.name+'_W')
        self.b = K.zeros((self.output_dim,),name=self.name+'_b')

        self.params = [self.W, self.b]

    def output(self, train=False):
        X = self._default_input(train)
        output = self.get_function('activation')(K.dot(X, self.W) + self.b)
        return output
    
    
    
class Dropout(Layer):
    '''Apply Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.p = p

        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += []
        self.set_output('output', self.output, self._default_output_mask)

    def output(self, train=False):
        X = self._default_input(train)
        if self.p > 0.:
            if train:
                X = K.dropout(X, level=self.p)
        return X


class Activation(Layer):
    '''Apply an activation function to an output.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self):
        super(Activation, self).__init__()
        
        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += ['activation']
        self.set_output('output', self.output, self._default_output_mask)

    def output(self, train=False):
        X = self._default_input(train)
        return self.get_function('activation')(X)

class Output(Layer):
    '''A layer for output
    '''
    def __init__(self):
        super(Output, self).__init__()
        
        self.input_names += ['input']
        self.output_names += []
        self.function_names += []
        
    def output(self, train=False):
        return self._default_input(train) 
    
    def output_mask(self, train=False):
        return self._default_input_mask(train)
    
class Lambda(Layer):
    '''A layer perform a given function without state 
    # input names += ['input_' + arg[0], 'input_' + arg[1]]
    # all the output mask = first input mask
    '''
    
    def __init__(self, function, return_names):
        super(Lambda, self).__init__()
        import inspect
        self.arg_names= ['input_' + arg for arg in inspect.getargspec(function)[0]]
        self.return_names = return_names
        
        self.input_names += self.arg_names
        self.output_names += self.return_names
        self.function_names += ['function']
        
        self.set_function('function', function)
        for return_name in return_names:
            self.set_output(return_name, self.get_output_function(return_name), self.first_input_mask)
           
    def get_output_function(self, return_name):
        return lambda train=False:self.output(return_name,train)
    
    def output(self, return_name, train=False):
        outputs = self.get_function('function')(*[self.get_input(arg_name) for arg_name in self.arg_names])
        return outputs[self.return_names.index(return_name)]
        
    def first_input_mask(self, train=False):
        if len(self.arg_names) >= 1:
            return self.get_input_mask(self.arg_names[0], train)    
        else:
            return self._no_output_mask(train)
        
class SimpleLambda(Layer):
    def __init__(self, function):
        super(SimpleLambda, self).__init__()
        
        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += ['function']
        self.set_output('output', self.output, self._default_output_mask)
        self.set_function('function', function)

    def output(self, train=False):
        X = self._default_input(train)
        return self.get_function('function')(X)    
        
        
class RepeatVector(Layer):
    '''Repeat the input n times.

    # Input shape
        2D tensor of shape `(nb_samples, input_dim)`.

    # Output shape
        3D tensor of shape `(nb_samples, n, input_dim)`.

    # Arguments
        n: integer, repetition factor.
    '''
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n
        
        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += []
        self.set_output('output', self.output, self._no_output_mask)

    def output(self, train=False):
        X = self._default_input(train)
        return TU.repeat(X, self.n).dimshuffle((1, 0, 2))
        
  
class TimeDistributedDense(Layer):
    '''Apply a same Dense layer for each dimension[1] (time dimension) input.
    Especially useful after a recurrent network with 'return_sequence=True'.

    # Input shape
        3D tensor with shape `(nb_sample, input_length, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_sample, input_length, output_dim)`.
    '''
    def __init__(self, input_length, input_dim, output_dim, name='TimeDistributedDense'):
        super(TimeDistributedDense, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        
        self.input_names += ['input']
        self.output_names += ['output']
        self.function_names += ['init', 'activation']
        self.set_output('output', self.output, self._no_output_mask)
        

    def build(self):
        self.W = self.get_function('init')((self.input_dim, self.output_dim), name=self.name+'_W')
        self.b = K.zeros((self.output_dim,),name=self.name+'_b')

        self.params = [self.W, self.b]

    def output(self, train=False):
        input = self._default_input(train) # (nb_sample, input_length, input_dim)
        X = input.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
        def step(x, W, b):
            output = K.dot(x, W) + b
            return output
        
        outputs, _ = theano.scan(step,
                                sequences= X,
                                outputs_info=[],
                                non_sequences=[self.W, self.b],
                                strict=True)
             
        outputs = self.get_function('activation')(outputs) #(input_length, nb_samples, output_dim)
        return outputs.dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    