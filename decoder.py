from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        pos[0]='<ROOT>'
        words[0]='<ROOT>'
        words.append('<NULL>')
        pos.append('<NULL>')
        pos=[i if "<" in i else "<"+i+">" for i in pos]

        while state.buffer: 
            n=3-len(state.stack)
            if n>0:
                s1=state.stack[::-1]+n*[len(words)-1]
            else:
                s1=state.stack[-3:][::-1]
            n=3-len(state.buffer)
            if n>0:
                s2=state.buffer[::-1]+n*[len(words)-1]
            else:
                s2=state.buffer[-3:][::-1]  
            s=s1+s2
            
            s=[words[i] if words[i] in self.extractor.word_vocab else pos[i] for i in s]
            s=[self.extractor.word_vocab[i] if i in self.extractor.word_vocab else 2 for i in s]
            pred=self.model.predict(np.reshape(s,(1,6)))[0]
            action=np.argmax(pred)
            if len(state.stack)==0:
                action=0
            if len(state.buffer)==1 and len(state.stack)>0 and action==0:  
                pred[0]=0
                action=np.argmax(pred)
            if s[0]==3:
                pred=[pred[i] if not 'left' in self.output_labels[i][0] else 0 for i in range(91)]
                action=np.argmax(pred)
            action=self.output_labels[action]
            if action[0]=='shift':
                state.stack.append(state.buffer[-1])
                del state.buffer[-1]
            elif action[0]=='right_arc':
                state.deps.add((state.stack[-1],state.buffer[-1],action[1]))
                if action[1]=='root':
                    del state.stack[-1]
                    del state.buffer[-1]
                else:
                    state.buffer[-1]=state.stack[-1]
                    del state.stack[-1]
            else:
                state.deps.add((state.buffer[-1],state.stack[-1],action[1]))
                del state.stack[-1]


        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
#python evaluate.py ./model.h5 ./data/dev.conll
        
