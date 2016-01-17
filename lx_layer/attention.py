'''
@author: Xiang Long
'''
from keras import backend as K
from layer import Layer
from theano import tensor as T
import numpy
import util.theano_util as TU 
import theano

class AttentionLSTM(Layer):
    '''Long-Short Term Memory with soft Attention.
    # Input shape
        3D tensor with shape `(nb_samples, input_length, input_dim)`.

    # Output shape
        output_sequences: 3D tensor with shape `(nb_samples, input_length, output_dim)`.
        output: 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, input_length, input_dim, output_dim, context_dim, attention_hidden_dim, name='AttentionLSTM', truncate_gradient=-1, go_backwards=False):
        super(AttentionLSTM, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.name=name
        self.truncate_gradient = truncate_gradient
        self.go_backwards = go_backwards
        
        self.input_names += ['input_context', 'input_single', 'input_sequences']
        self.output_names += ['output_last', 'output_sequences']
        self.function_names += ['init', 'inner_init', 'forget_bias_init', 'inner_activation', 'activation', 'attention_activation']
        self.set_output('output_last', self.output_last, self._no_output_mask)
        self.set_output('output_sequence', self.output_sequence, self._default_output_mask)
        
    def build(self):
        f_init = self.get_function('init')
        f_inner_init = self.get_function('inner_init')
        f_forget_bias_init = self.get_function('forget_bias_init')
        
        '''
        Attention hidden dense and projector
        '''
        self.W_h_att = f_init((self.output_dim, self.attention_hidden_dim), name=self.name + '_W_h_att')
        self.W_ctx_att = f_init((self.context_dim, self.attention_hidden_dim), name=self.name + '_W_ctx_att')
        self.b_att =  K.zeros((self.attention_hidden_dim,), name=self.name + '_b_att')
        self.w_att_prj = f_init((self.attention_hidden_dim, 1), name=self.name + '_w_att_prj')

        ''' 
        LSTM {W: x, U: h, V: weighted context}
        '''
        # numpy matrixes
        W_i = f_init((self.input_dim, self.output_dim), name=self.name + '_W_i').get_value()
        V_i = f_init((self.context_dim, self.output_dim), name=self.name + '_V_i').get_value()
        U_i = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_i').get_value()
        b_i = K.zeros((self.output_dim,), name=self.name + '_b_i').get_value()

        W_f = f_init((self.input_dim, self.output_dim), name=self.name + '_W_f').get_value()
        V_f = f_init((self.context_dim, self.output_dim), name=self.name + '_V_f').get_value()
        U_f = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_f').get_value()
        b_f = f_forget_bias_init((self.output_dim,), name=self.name + '_b_f').get_value()

        W_c = f_init((self.input_dim, self.output_dim), name=self.name + '_W_c').get_value()
        V_c = f_init((self.context_dim, self.output_dim), name=self.name + '_V_c').get_value()
        U_c = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_c').get_value()
        b_c = K.zeros((self.output_dim,), name=self.name + '_b_c').get_value()

        W_o = f_init((self.input_dim, self.output_dim), name=self.name + '_W_o').get_value()
        V_o = f_init((self.context_dim, self.output_dim), name=self.name + '_V_o').get_value()
        U_o = f_inner_init((self.output_dim, self.output_dim), name=self.name + '_U_o').get_value()
        b_o = K.zeros((self.output_dim,), name=self.name + '_b_o').get_value()
        
        # theano variables
        self.W = theano.shared(numpy.concatenate([W_i, W_f, W_c, W_o], axis=1), name=self.name + '_W' , strict=False)
        self.V = theano.shared(numpy.concatenate([V_i, V_f, V_c, V_o], axis=1), name=self.name + '_V' , strict=False)
        self.U = theano.shared(numpy.concatenate([U_i, U_f, U_c, U_o], axis=1), name=self.name + '_U' , strict=False)
        self.b = theano.shared(numpy.concatenate([b_i, b_f, b_c, b_o]), name=self.name + '_b' , strict=False)
        
        self.params = [self.W, self.V, self.U, self.b, self.W_h_att, self.W_ctx_att, self.b_att, self.w_att_prj]
    
    def _slice(self, P, i, dim):
        return P[:, i*dim:(i+1)*dim]

    def step(self, mask_t, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        prj_ctx = T.flatten(T.dot(preprj, w_att_prj))  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(x_t, W) + T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + b
        i = f_inner_activation(self._slice(preact, 0, self.output_dim))
        f = f_inner_activation(self._slice(preact, 1, self.output_dim))
        c = f * c_tm1 + i * f_activation(self._slice(preact, 3, self.output_dim))
        o = f_inner_activation(self._slice(preact, 2, self.output_dim))
        h = o * f_activation(c)
        # mask
        h = T.switch(mask_t, h, 0. * h)
        c = T.switch(mask_t, c, 0. * c)
        return h, c    

    def step_no_mask(self, x_t, h_tm1, c_tm1, ctx, att_ctx, W, V, U, b, W_h_att, w_att_prj):
        f_activation = self.get_function('activation')
        f_inner_activation = self.get_function('inner_activation')
        f_attention_activation = self.get_function('attention_activation')
        
        # attention
        h_att = T.dot(h_tm1, W_h_att) # (nb_sample, input_dim) dot (input_dim, attention_hidden_dim) -> (nb_sample, attention_hidden_dim)
        preprj = f_attention_activation(h_att[:,None,:] + att_ctx) #(nb_sample, nb_context, attention_hidden_dim)
        prj_ctx = T.flatten(T.dot(preprj, w_att_prj), 2)  #(nb_sample, nb_context)
        alpha = T.nnet.softmax(prj_ctx) #(nb_sample, nb_context)
        weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (nb_sample, context_dim)
        
        # LSTM
        preact = T.dot(x_t, W) + T.dot(weighted_ctx, V) + T.dot(h_tm1, U) + b
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
        
        context = self.get_input('context', train)  # (nb_samples, nb_context, context_dim)
        att_ctx = T.dot(context, self.W_ctx_att) + self.b_att  # (nb_samples, nb_context, context_dim) + (context_dim,)
 
        h_0 = T.zeros((X.shape[1], self.output_dim), X.dtype)  # (nb_samples, output_dim)
        #h_0 = self._get_initial_state(X)
        c_0 = T.zeros((X.shape[1], self.output_dim), X.dtype)  # (nb_samples, output_dim)

        if mask:
            revals, _ = theano.scan( self.step,
                                    sequences=[mask, X],
                                    outputs_info=[h_0, c_0],
                                    non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.w_att_prj],
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True)
        else:
            revals, _ = theano.scan( self.step_no_mask,
                                     sequences=[X],
                                    outputs_info=[h_0, c_0],
                                    non_sequences=[context, att_ctx, self.W, self.V, self.U, self.b, self.W_h_att, self.w_att_prj],
                                    truncate_gradient=self.truncate_gradient,
                                    go_backwards=self.go_backwards,
                                    strict=True)
            
        return revals[0] #(input_length, nb_samples, output_dim)
    
    def output_sequence(self, train=False):
        return self.output_h_vals(train).dimshuffle((1, 0, 2)) # (nb_sample, input_length, output_dim)
    
    def output_last(self, train=False):
        return self.output_h_vals(train)[-1] # (nb_sample, output_dim)
    