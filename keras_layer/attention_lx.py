import numpy as np
import theano_backend_lx as LX

from keras import backend as K
from keras import activations, initializations
from keras.layers.core import MaskedLayer
from keras.layers.recurrent import Recurrent
from networkx.algorithms.bipartite.projection import projected_graph
from astropy.constants.si import alpha


class RNNAttention(Recurrent):
    '''Abstract base class for soft RNN attention layers.
    Do not use in a model -- it's not a functional layer!
    
    All RNN attention layers (___, ___, ___) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.
    '''

    def __init__(self, truncate_gradient=-1, **kwargs):
        if K._BACKEND == 'tensorflow':
            raise Exception('RNNAttention is Theano-only for the time being.')
        self.contexts = []
        self.truncate_gradient = truncate_gradient
        super(RNNAttention, self).__init__(**kwargs)
                        
    def add_context(self, context):
        '''set the context of the attention layer. 

        # Arguments
            context: the context layer.
        '''
        self.contexts.append(context)
        
    def attention_step(self, x, args):
        ''' calculate output with context

        # Arguments
            args = [states, contexts, parameters]
        '''
        raise NotImplementedError
    
    def get_standard_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states
    
    def get_other_initial_states(self, X, *args):
        ''' get the initial states

        # Return
            a list of initial state corresponds to self.states except first
            nb_standard standard states.
        '''
        raise NotImplementedError
            
    def get_initial_states(self, X, *args):
        return self.get_standard_initial_states(X) + self.get_other_initial_states(X, args)
    
    def get_output(self, train = False, get_tuple = False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        
        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, other_outputs, states = LX.rnn(self.attention_step, X, initial_states, self.contexts,
                                              truncate_gradient=self.truncate_gradient,
                                              go_backwards=self.go_backwards,
                                              masking=masking)
        self.other_outputs = other_outputs
        
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output
        
    def get_other_outputs(self):
        if hasattr(self, "other_outputs"):
            return self.other_outputs
        else:
            raise Exception('Run get_output() first.')
        
        
    def get_config(self):
        config = {"truncate_gradient": self.truncate_gradient,
                  'contexts': [context.get_config() for context in self.contexts]}
        base_config = super(RNNAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
class SimpleRNNAttention(RNNAttention):  
    '''Fully-connected RNN with soft attention.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
    '''
    def __init__(self, hidden_dim, output_dim, context,
                 projected_context_dim = None,
                 init='glorot_uniform', inner_init='orthogonal',
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context = context;
        self.projected_context_dim = projected_context_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        
        self.context_dim = K.shape(context)[1]
        if not self.projected_context_dim:
            self.projected_context_dim = self.context_dim
            
        super(SimpleRNNAttention, self).__init__(**kwargs)

    def build(self):
        
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        ''' W_ctx_prj, b_ctx_prj, W_h_prj, 
            w_prj_att, b_att,
            W_x_h, U_h_h, W_ctx_h, b_h
            W_x_p, W_h_p, W_ctx_p, b_p].
            No implement.
        
        '''
                
        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def attention_step(self, x, args):
        ''' Attention step function
        #Arguments
            args: [h_,
                    context, projected_context, 
                    W_h_prj, 
                    w_prj_att, b_att,
                    W_x_h, U_h_h, W_ctx_h, b_h
                    W_x_p, W_h_p, W_ctx_p, b_p].
                h_: (batch_size, dim_hidden)
                context: (batch_size, nb_context, dim_context)
                projected_context: (batch_size, nb_context, dim_projected_context)
                    projected_context = context dot W_ctx_prj + b_ctx_prj
                    calculated before step.
                W_h_prj: (dim_hidden, dim_projected_context)
                w_prj_att: (dim_projected_context, 1)
                b_att: (1,)
                W_x_h: (dim_embedding, dim_hidden)
                U_h_h: (dim_hidden, dim_hidden)
                W_ctx_h: (dim_context, dim_hidden)
                b_h: (dim_hidden,)                
        '''
        assert len(args) == 1 + len(self.contexts) + len(self.params)
        [h_, context, projected_context, 
         W_h_prj, 
         w_prj_att, b_att,
         W_x_h, U_h_h, W_ctx_h, b_h] = args
        
        projected = K.expand_dims(K.dot(h_, W_h_prj), 1) + projected_context
        e = K.dot(K.tanh(projected), w_prj_att) + b_att
        alpha = K.softmax(K.flatten(e))
        weighted_context = K.sum((context * K.expand_dims(alpha)), 1)
        pre_act = K.dot(x, W_x_h) + K.dot(h_, U_h_h) + K.dot(weighted_context, W_ctx_h) + b_h
        h = K.sigmoid(pre_act)
        
    
        return h, [alpha, weighted_context], [h]
        
        
    def get_other_initial_states(self, X, *args):
        ''' get the initial states

        # Return
            a list of initial state corresponds to self.states except first
            nb_standard standard states.
        '''
        raise NotImplementedError
        

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(SimpleRNNAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    
    
    
    
    
    
    
    
      
    
    