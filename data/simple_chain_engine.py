import numpy as np

class SimpleChainEngine(object):
    
    def __init__(self, chars):
        self.chars = chars
        self.cnt = len(chars)
            
    def get_start(self):
        return np.random.randint(self.cnt - 1)
    
    def get_chain(self, start):
        s = ''
        for i in range(start + 1, self.cnt):
            s += self.chars[i]
        return s
            
    def get_data(self):
        start = self.get_start()
        return self.chars[start], self.get_chain(start)
    
    def get_dataset(self, size):
        starts = []
        chains = []
        for _ in range(size):
            q, a = self.get_data()
            starts.append(q)
            chains.append(a)
        return starts, chains
    
    def get_character_set(self):
        return self.chars
        
            
if __name__ == '__main__':
    engine = SimpleChainEngine('0123456789abcdef')
    s, c = engine.get_data()
    print "%s -> %s" %(s,c)
    ss, cs = engine.get_dataset(10)
    for (s, c) in zip(ss, cs):
        print "%s -> %s" %(s,c)
    
    print engine.get_character_set() 