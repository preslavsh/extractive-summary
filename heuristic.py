import math

import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from utils import load_data, calculate_vec, rougeL

document_list, summaries = load_data(5)

sent_tokenizer = PunktSentenceTokenizer()
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'ms', 'u.s', 'rep'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

size = len(document_list)

scores = np.array([])
rouges = np.array([])

for i in range(size):
    print('=====doc======')
    print(i)
    doc = document_list[i]
    sentenses = sentence_splitter.tokenize(doc)
    print(sentenses)
    log_size = np.log10(len(sentenses))
    max_items = int(log_size) if log_size >= 1.0 else 1
    summary = ''
    if len(sentenses) == 0:
        summary = ''
    elif max_items == 1:
        summary = sentenses[0]
    elif max_items == 2:
        summary = "".join(sentenses[:2])
    else:
        summary = "".join(sentenses[:3])
    score = calculate_vec(summary, summaries[i])
    rougeL_score = rougeL(summary, summaries[i])

    print('=====real=======')
    print(summaries[i])
    print('======mine======')
    print(summary)
    print('======vec=======')
    print(score)
    if math.isnan(score) is False:
        scores = np.append(scores, score)
    print('====rouge1==========')
    print(rougeL_score)
    if math.isnan(rougeL_score) is False:
        rouges = np.append(rouges, rougeL_score)
    print('====================')


print('=====cos avg and median=======')
scores = scores[np.logical_not(np.isnan(scores))]
print(scores)
print(np.average(scores))
print(np.median(scores))
print('=====rouge avg and median=======')
rouges = rouges[np.logical_not(np.isnan(rouges))]
print(rouges)
print(np.average(rouges))
print(np.median(rouges))
print('====================')
