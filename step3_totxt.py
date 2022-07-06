import sys

sys.path.append('/home/amax/zzhaoao/BookComprehension/pipeline/questions/pattern')

import os
from copy import deepcopy
import codecs
from nltk.tokenize.moses import MosesDetokenizer
from conllu import parse
from rule import Question, AnswerSpan
import pattern


INPUT_PATH = '/home/amax/zzhaoao/BookComprehension/pipeline/questions/new_parsed/conll2'
OUTPUT_PATH = '/home/amax/zzhaoao/BookComprehension/pipeline/questions/new_parsed/inference'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def qa2d(idx):
    # q = Question(deepcopy(examples[idx].tokens))
    q = Question(deepcopy(examples[idx]))
    if not q.isvalid:
        print("Question {} is not valid.".format(idx))
        return ''
    # a = AnswerSpan(deepcopy(examples[str(idx)+'_answer'].tokens))
    a = AnswerSpan(deepcopy(examples[str(idx)+'_answer']))
    if not a.isvalid:
        print("Answer span {} is not valid.".format(idx))
        return ''
    q.insert_answer_default(a)
    return detokenizer.detokenize(q.format_declr(), return_str=True)

def print_sentence(idx):
    return detokenizer.detokenize([examples[idx][i]['form'] for i in range(len(examples[idx]))], return_str=True)

detokenizer = MosesDetokenizer()
    
files = [f for f in os.listdir(INPUT_PATH) if os.path.isfile(os.path.join(INPUT_PATH, f))]
for file in files:
    results = ''
    # if file=='ibm_data_QA-pairs-second-round_scottish-fairybook_the-wee-bannock-questions_10.conll':
    #     print(file)
    with codecs.open(os.path.join(INPUT_PATH, file), 'r', encoding='utf-8') as f, open(os.path.join(OUTPUT_PATH, file.replace('.conll', '.txt')), 'w') as wf:
        print(file)
        conllu_file = parse(f.read())
        
        # Creating dict
        ids = range(int(len(conllu_file)/2))
        examples = {}
        count = 0
        for i, s in enumerate(conllu_file):
            if i % 2 == 0:
                examples[ids[count]] = s
            else:
                examples[str(ids[count])+'_answer'] = s
                count +=1
                
        total = int(len(examples.keys())/2)
        print("Transforming {} examples.".format(total))
        for i in range(total):
            out = qa2d(i)
            # print(print_sentence(i))
            # if out != '':
            #     print(out)
            # print('----------')
            results += out + '\n\n'
        wf.write(results)
