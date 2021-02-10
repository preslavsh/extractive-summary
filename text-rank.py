import math

import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from utils import calculate_vec, load_data, rougeL

# Influenced by PageRank algorithm, these methods represent documents as a connected graph,
# where sentences form the vertices and edges between the sentences indicate how similar
# the two sentences are. The similarity of two sentences is measured with the help of
# cosine similarity with TFIDF weights for words and if it is greater than a certain threshold,
# these sentences are connected. This graph representation results in two outcomes:
# the sub-graphs included in the graph create topics covered in the documents,
# and the important sentences are identified. Sentences that are connected to
# many other sentences in a sub-graph are likely to be the center of the graph and
# will be included in the summary Since this method do not need language-specific
# linguistic processing, it can be applied to various languages [43].
# At the same time, such measuring only of the formal side of the sentence
# structure without the syntactic and semantic information limits the application of the method.

documents_list, summaries = load_data(5)

# Add connection counts to the
def get_connections_dictionary(sentenses):
    size = len(sentenses)
    d = dict.fromkeys(sentenses, 0)
    for i in range(size):
        current_sentence = sentenses[i]
        # We are searching only for one match
        # We are not supporting multiple max measures
        # or n-references to
        max_measured = 0
        max_index = 0
        for j in range(size):
            if i != j:
                measured_against = sentenses[j]
                # TODO: Use tf-idf
                cos = calculate_vec(current_sentence, measured_against)
                if cos > max_measured:
                    max_measured = cos
                    max_index = j

        # We are adding count to the max score for measured sentence
        # because the current is pointing to the measured
        d[sentenses[max_index]] += 1
    return d


def get_mine_summary(d, sentences):
    size = len(sentences)
    # Custom measure max sentences to get for summary
    max_size_float = np.log10(size)
    max_items = int(max_size_float) if max_size_float >= 1.0 else 1
    values = list(d.values())
    enumerated = [(v, i) for i, v in enumerate(values)]
    enumerated.sort(key=lambda tup: tup[0])
    enumerated.reverse()
    summary_elements = enumerated[0:max_items]
    return "".join([ sentenses[el[1]] for el in summary_elements])


sent_tokenizer = PunktSentenceTokenizer()
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'ms', 'u.s', 'rep'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

scores = np.array([])
rouges = np.array([])

for i in range(len(documents_list)):
    doc = documents_list[i]
    sentenses = sentence_splitter.tokenize(doc)
    d = get_connections_dictionary(sentenses)
    summary = get_mine_summary(d, sentenses)
    score = calculate_vec(summary, summaries[i])
    rouge1_score = rougeL(summary, summaries[i])
    print('=====doc======')
    print(doc)
    print(i)
    print('=====real=======')
    print(summaries[i])
    print('======mine======')
    print(summary)
    print('======vec=======')
    print(score)
    if math.isnan(score) is False:
        scores = np.append(scores, score)
    print('====rouge1==========')
    print(rouge1_score)
    if math.isnan(rouge1_score) is False:
        rouges = np.append(rouges, rouge1_score)
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


