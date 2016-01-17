from keras.layers.core import Layer
from keras import backend as K


class Shift(Layer):
    '''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!Can not work!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    ''' Shift a input sequence many steps to get a output sequence.
    # Arguments
        sequence_layer: the layer whose output sequence we want to
            shift when testing.
        initial_value: initial_value of the 0 position of sequence.
            (nb_sample, input_dim) or (nb_sample, 1, input_dim)
    '''
    def __init__(self, sequence_layer, initial_layer, **kwargs):
        self.sequence_layer = sequence_layer
        self.initial_layer = initial_layer
        super(Shift, self).__init__(**kwargs)
        
    def supports_masked_input(self):
        return True

    def get_input_mask(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output_mask(train)
        else:
            return None
    
    def get_output_mask(self, train=False):   
        ''' Shift the mask
            the mask is (nb_samples, nb_timesteps)        
            with a one for every unmasked datapoint,
            and a zero for every masked one
        '''
        if K._BACKEND == 'tensorflow':
            raise Exception('Masking is Theano-only for the time being.')
        
        if train:
            input_mask = self.get_input_mask(train)
        else:
            input_mask = self.sequence_layer.get_output_mask(train)
            
        if not input_mask:
            return None        
        head = K.ones((K.shape(input_mask)[0], 1))        
        output_mask = K.concatenate((head, input_mask[:,:-1]), axis=1)
        return output_mask          

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return input_shape
        
    def get_output(self, train=False):
        '''Shift the sequence
            the input X is (nb_samples, nb_timesteps, input_dim)
        '''
        if train:
            X = self.get_input(train)
        else:
            X = self.sequence_layer.get_output() 
            
        head = K.expand_dims(K.flatten(self.initial_layer.get_output()), 1)
        output = K.concatenate((head, X[:,:-1,:]) ,axis=1)
        
        return output


    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'sequence_layer': self.sequence_layer.get_config(),
                  'initial_layer': self.initial_layer.get_config()}
        base_config = super(Shift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    