import numpy as np

class CharacterDataEngine(object):
    '''
    Given a set of string:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.dim = len(self.chars) + 1;
        self.char_indices = dict((c, i + 1) for i, c in enumerate(self.chars))
        self.indices_char = dict((i + 1, c) for i, c in enumerate(self.chars))
        self.indices_char[0] = ''
        self.maxlen = maxlen
    
    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def encode(self, string, maxlen=None, invert=False):
        maxlen = maxlen if maxlen else self.maxlen
        vectors = np.zeros((maxlen, self.dim), dtype=np.bool)
        for i, c in enumerate(string):
            vectors[i, self.char_indices[c]] = True
        for i in range(len(string), maxlen):
            vectors[i, 0] = True
        if invert:
            return vectors[::-1]
        else:
            return vectors

    def decode(self, vectors, calc_argmax=True, invert=False):
        if calc_argmax:
            vectors = vectors.argmax(axis=-1)
        if invert:
            vectors = vectors[::-1]
        return ''.join(self.indices_char[v] for v in vectors)
    
    def encode_dataset(self, strings, maxlen=None, invert=False):
        maxlen = maxlen if maxlen else self.maxlen
        datas = np.zeros((len(strings), maxlen, self.dim), dtype=np.bool)
        for i, sentence in enumerate(strings):
            datas[i] = self.encode(sentence, maxlen, invert)
        return datas
    
    def decode_dataset(self, datas, calc_argmax=True, invert=False):
        strings = []
        for vectors in datas:
            strings.append(self.decode(vectors, calc_argmax, invert))
        return strings
    
    def get_dim(self):
        return self.dim
    
if __name__ == '__main__':
    engine = CharacterDataEngine('0123456789+', 10)
    s = '0123+89'
    v = engine.encode(s)
    print v
    r = engine.decode(v)
    print r
    
    ss = ['12+34', '7890+54321', '0+0']
    d = engine.encode_dataset(ss, 12)
    print d
    rs = engine.decode_dataset(d)
    print rs
    
    
    