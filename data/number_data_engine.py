'''
Created on 2016/01/09

@author: Xiang Long
'''
import numpy as np

class NumberDataEngine(object):
    '''
    classdocs
    '''
    num_chars = '0123456789'
    operator_func_dict = {'+':lambda x,y:x+y,
                          '-':lambda x,y:x-y,
                          '*':lambda x,y:x*y
                          }
    
    seen = set()

    def __init__(self, min_digits=1, max_digits=3, sort=True, operator_char='+', operator_func = None):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.sort = sort
        self.operator_char = operator_char
        self.operator_func = operator_func
        
        if not self.operator_func:
            if (operator_char in self.operator_func_dict):
                self.operator_func = self.operator_func_dict[operator_char]
            else:
                raise Exception('Operator function of "%s" has not been implemented.'% operator_char)
            
    def get_number(self):
        if self.min_digits == 1:
            return np.random.randint(10**self.max_digits)
        else:
            return np.random.randint(10**self.min_digits, 10**self.max_digits)

            
    def get_data(self):
        while 1:
            a, b = self.get_number(), self.get_number()  
            if self.sort:   
                key = tuple(sorted((a, b))) # Skip any such that A+B == A+B or A+B == B+A (hence the sorting)
            else:
                key = (a,b) # Skip any addition questions we've already seen 
            if key not in self.seen:
                self.seen.add(key)
                question = "%d%s%d" %(a, self.operator_char, b)
                answer = str(self.operator_func(a,b))
                return question, answer
    
    def get_dataset(self, size):
        questions = []
        answers = []   
        for _ in range(size):
            q, a = self.get_data()
            questions.append(q)
            answers.append(a)
        return questions, answers
    
    def get_character_set(self):
        return '0123456789' + self.operator_char
        
            
if __name__ == '__main__':
    engine = NumberDataEngine(min_digits=1, max_digits=1, sort=False, operator_char='+')
    q, a = engine.get_data()
    print "%s = %s" %(q,a)
    qs, anss = engine.get_dataset(10)
    for (q, a) in zip(qs, anss):
        print "%s = %s" %(q,a)
    
    print 
    
    def mod (x, y):
        if y != 0:
            return x%y
        else:
            return 0
    
    engine = NumberDataEngine(min_digits=1, max_digits=3, sort=True, operator_char='%', operator_func=mod)
    q, a = engine.get_data()
    print "%s = %s" %(q,a)
    qs, anss = engine.get_dataset(10)
    for (q, a) in zip(qs, anss):
        print "%s = %s" %(q,a)
    
    print engine.get_character_set()