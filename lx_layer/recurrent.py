'''
@author: Xiang Long
'''
from keras import backend as K
from layer import Layer
from theano import tensor as T
import numpy
import util.theano_util as TU 
import theano

class RNN(Layer):
    '''Fully-connected RNN where the output is to fed back to input.
    # Input shape
        3D tensor with shape `(nb_samples, input_length, input_dim)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, name='RNN', truncate_gradient=-1, go_backwards=False):
        super(RNN, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.input_names += ['input_single', 'input_sequences']
        self.output_names += ['output_last', 'output_sequences']
        self.function_names += ['init', 'inner_init', 'activation']
        self.set_output('output_last', self.output_last, self._no_output_mask)
        self.set_output('output_sequence', self.output_sequence, self._default_output_mask)
        
    def build(self):
        self.W = self.get_function('init')((self.input_dim, self.output_dim), name=self.name + '_W')
        self.U = self.get_function('inner_init')((self.output_dim, self.output_dim), name=self.name + '_U')
        self.b = K.zeros((self.output_dim,), name=self.name + '_b')
        self.params = [self.W, self.U, self.b]
    
    def step(self, mask_t, x_t, h_tm1, W, U, b):
        h_t = self.get_function('activation')(T.dot(x_t, W) + T.dot(h_tm1, U) + b)
        # mask
        h_t = T.switch(mask_t, h_t, 0. * h_t)
        return h_t
    
    def step_no_mask(self, x_t, h_tm1, W, U, b):
        h_t = self.get_function('activation')(T.dot(x_t, W) + T.dot(h_tm1, U) + b)
        return h_t
    
    def _get_initial_state(self, X): # X (input_length, nb_sample, input_dim)
        # build an all-zero tensor of shape (nb_samples, output_dim)
        initial_state = K.zeros_like(X)  # (input_length, nb_sample, input_dim)
        initial_state = K.sum(initial_state, axis=0)  # (nb_samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (nb_samples, output_dim)
        return initial_state
    
    def output_h_vals(self, train=False):
        if self.inputs_dict.has_key('input_single'):
            input = self.get_input('input_single', train) #(nb_sample, input_dim)
            X = TU.repeat(input, self.input_length) # (input_length, nb_sample, input_dim)    
            mask = None
        else:
            input = self.get_input('input_sequence', train)  # (nb_sample, input_length, input_dim)
            X = input.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
            mask = self.get_input_mask('input_sequence',train) # (nb_sample, input_length)    
            if mask:
                mask = T.cast(mask, dtype='int8').dimshuffle((1, 0, 'x')) # (input_length, nb_sample, 1)

        #h_0 = T.zeros((X.shape[1], self.output_dim), X.dtype)  # (nb_samples, output_dim)
        h_0 = self._get_initial_state(X)

        if mask:
            h_vals, _ = theano.scan( self.step,
                                            sequences=[mask, X],
                                            outputs_info=h_0,
                                            non_sequences=[self.W, self.U, self.b],
                                            truncate_gradient=self.truncate_gradient,
                                            go_backwards=self.go_backwards,
                                            strict=True)
        else:
            h_vals, _ = theano.scan( self.step_no_mask,
                                sequences=[X],
                                outputs_info=h_0,
                                non_sequences=[self.W, self.U, self.b],
                                truncate_gradient=self.truncate_gradient,
                                go_backwards=self.go_backwards,
                                strict=True)
            
        return h_vals #(input_length, nb_samples, output_dim)
    
    def output_sequence(self, train=False):
        return self.output_h_vals(train).dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_h_vals(train)[-1] # (nb_sample, output_dim)
    
    
class LSTM(Layer):
    '''Long-Short Term Memory.
    # Input shape
        3D tensor with shape `(nb_samples, input_length, input_dim)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, name='LSTM', truncate_gradient=-1, go_backwards=False):
        super(LSTM, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.input_names += ['input_single', 'input_sequences']
        self.output_names += ['output_last', 'output_sequences']
        self.function_names += ['init', 'inner_init', 'forget_bias_init', 'inner_activation', 'activation']
        self.set_output('output_last', self.output_last, self._no_output_mask)
        self.set_output('output_sequence', self.output_sequence, self._default_output_mask)
        
    def build(self):
        f_init = self.get_function('init')
        f_inner_init = self.get_function('inner_init')
        f_forget_bias_init = self.get_function('forget_bias_init')
         
        # numpy matrixes
        W_i = f_init((self.input_dim, self.output_dim), name=self.name + '_W_i').get_value()
        U_i = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_i').get_value()
        b_i = K.zeros((self.output_dim,), name=self.name + '_b_i').get_value()

        W_f = f_init((self.input_dim, self.output_dim), name=self.name + '_W_f').get_value()
        U_f = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_f').get_value()
        b_f = f_forget_bias_init((self.output_dim,), name=self.name + '_b_f').get_value()

        W_c = f_init((self.input_dim, self.output_dim), name=self.name + '_W_c').get_value()
        U_c = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_c').get_value()
        b_c = K.zeros((self.output_dim,), name=self.name + '_b_c').get_value()

        W_o = f_init((self.input_dim, self.output_dim), name=self.name + '_W_o').get_value()
        U_o = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_o').get_value()
        b_o = K.zeros((self.output_dim,), name=self.name + '_b_o').get_value()
        
        # theano variables
        self.W = theano.shared(numpy.concatenate([W_i, W_f, W_c, W_o], axis=1), name=self.name + '_W' , strict=False)
        self.U = theano.shared(numpy.concatenate([U_i, U_f, U_c, U_o], axis=1), name=self.name + '_U' , strict=False)
        self.b = theano.shared(numpy.concatenate([b_i, b_f, b_c, b_o]), name=self.name + '_b' , strict=False)
        self.params = [self.W, self.U, self.b]
    
    def _slice(self, P, i, dim):
        return P[:, i*dim:(i+1)*dim]

    def step(self, mask_t, x_t, h_tm1, c_tm1, W, U, b):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        preact = T.dot(x_t, W) + T.dot(h_tm1, U) + b

        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))

        h = o * f_activation(c)
        # mask
        h = T.switch(mask_t, h, 0. * h)
        c = T.switch(mask_t, c, 0. * c)
        return h, c    

    def step_no_mask(self, x_t, h_tm1, c_tm1, W, U, b):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        preact = T.dot(x_t, W) + T.dot(h_tm1, U) + b

        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))

        h = o * f_activation(c)
        return h, c
    
    def _get_initial_state(self, X): # X (input_length, nb_sample, input_dim)
        # build an all-zero tensor of shape (nb_samples, output_dim)
        initial_state = K.zeros_like(X)  # (input_length, nb_sample, input_dim)
        initial_state = K.sum(initial_state, axis=0)  # (nb_samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (nb_samples, output_dim)
        return initial_state
    
    def output_h_vals(self, train=False):
        if self.inputs_dict.has_key('input_single'):
            input = self.get_input('input_single', train) #(nb_sample, input_dim)
            X = TU.repeat(input, self.input_length) # (input_length, nb_sample, input_dim)    
            mask = None
        else:
            input = self.get_input('input_sequence', train)  # (nb_sample, input_length, input_dim)
            X = input.dimshuffle((1, 0, 2))  # (input_length, nb_sample, input_dim) 
            mask = self.get_input_mask('input_sequence',train) # (nb_sample, input_length)    
            if mask:
                mask = T.cast(mask, dtype='int8').dimshuffle((1, 0, 'x')) # (input_length, nb_sample, 1)

        h_0 = T.zeros((X.shape[1], self.output_dim), X.dtype)  # (nb_samples, output_dim)
        #h_0 = self._get_initial_state(X)
        c_0 = T.zeros((X.shape[1], self.output_dim), X.dtype)  # (nb_samples, output_dim)

        if mask:
            revals, _ = theano.scan( self.step,
                                            sequences=[mask, X],
                                            outputs_info=[h_0, c_0],
                                            non_sequences=[self.W, self.U, self.b],
                                            truncate_gradient=self.truncate_gradient,
                                            go_backwards=self.go_backwards,
                                            strict=True)
        else:
            revals, _ = theano.scan( self.step_no_mask,
                                sequences=[X],
                                outputs_info=[h_0, c_0],
                                non_sequences=[self.W, self.U, self.b],
                                truncate_gradient=self.truncate_gradient,
                                go_backwards=self.go_backwards,
                                strict=True)
            
        return revals[0] #(input_length, nb_samples, output_dim)
    
    def output_sequence(self, train=False):
        return self.output_h_vals(train).dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_h_vals(train)[-1] # (nb_sample, output_dim)
    