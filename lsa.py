# Latent Semantic Analysis is an algebraic-statistical method that extracts hidden semantic structures of words and sentences. It is an unsupervised approach that does not need any training or external knowledge. LSA uses the context of
# the input document and extracts information such as which words are used together and which common words are seen
# in different sentences. A high number of common words among sentences indicates that the sentences are semantically
# related. The meaning of a sentence is decided using the words it contains, and meanings of words are decided using the
# sentences that contains the words. Singular Value Decomposition, an algebraic method, is used to find out the interrelations between sentences and words. Besides having the capability of modelling relationships among words and sentences, SVD has the capability of noise reduction, which helps to improve accuracy. In order to see how LSA can
# represent the meanings of words and sentences.


# 3.1.1. Step 1
# Input matrix creation: an input document needs to be represented in a way that enables a computer to understand and
# perform calculations on it. This representation is usually a matrix representation where columns are sentences and rows
# are words/phrases. The cells are used to represent the importance of words in sentences. Different approaches can be
# used for filling out the cell values. Since all words are not seen in all sentences, most of the time the created matrix is
# sparse.
# The way in which an input matrix is created is very important for summarization, since it affects the resulting
# matrices calculated with SVD. As already mentioned, SVD is a complex algorithm and its complexity increases with
# the size of input matrix, which degrades the performance. In order to reduce the matrix size, rows of the matrix, i.e. the
# words, can be reduced by approaches like removing stop words, using the roots of words only, using phrases instead of
# words and so on. Also, cell values of matrix can change the results of SVD. There are different approaches to filling out
# the cell values. These approaches are as follows.
# • Frequency of word: the cell is filled in with the frequency of the word in the sentence.
# • Binary representation: the cell is filled in with 0/1 depending on the existence of a word in the sentence.
# • Tf-idf (Term Frequency-Inverse Document Frequency): the cell is filled in with the tf-idf value of the word.
# A higher tf-idf value means that the word is more frequent in the sentence but less frequent in the whole
# document. A higher value also indicates that the word is much more representative for that sentence than
# others.
# • Log entropy: the cell is filled in with the log-entropy value of the word, which gives information on how informative the word is in the sentence.
# • Root type: the cell is filled in with the frequency of the word if its root type is a noun, otherwise the cell value
# is set to 0.
# • Modified Tf-idf: this approach is proposed in Ozsoy et al. [3], in order to eliminate noise from the input matrix.
# The cell values are set to tf-idf scores first, and then the words that have scores less than or equal to the average
# of the row are set to 0.

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from utils import load_data, get_stemmed, calculate_vec, rougeL

document_list, summaries = load_data(5)


def get_idf_dataframe(sentenses):
    document_frequency = defaultdict(lambda: 0)
    sentenses_stemmed = defaultdict()
    for sent in sentenses:
        stemmed_tokens = get_stemmed(sent)
        sentenses_stemmed[sent] = stemmed_tokens
        for _word in set(stemmed_tokens):
            document_frequency[_word] += 1

    idf_df = pd.DataFrame(list(document_frequency.items()), columns=['word', 'doc_freq'])
    idf_df['idf'] = np.log10(len(sentenses)/idf_df['doc_freq'])
    idf_df.sort_values(by=['idf'], inplace=True)
    return idf_df, sentenses_stemmed


def tfidf(word, sent, idf_df, sentenses_stemmed):
    sent_tokens = sentenses_stemmed[sent]
    count = sent_tokens.count(word)
    tf = (1 + np.log10(count)) if count != 0 else 1
    inner_idf = idf_df[idf_df['word'] == word].iloc[0]['idf']
    return  tf * inner_idf


# 3.2.1. Gong and Liu (2001). The algorithm of Gong and Liu [4] is one of the main studies conducted in LSA-based text
# summarization. After representing the input document in the matrix and calculating SVD values, VT matrix, the matrix
# Figure 1. LSA can represent the meaning of words and sentences.
# of extracted concepts × sentences is used for selecting the important sentences. In VT matrix, row order indicates the
# importance of the concepts, such that the first row represents the most important concept extracted. The cell values of
# this matrix show the relation between the sentence and the concept. A higher cell value indicates that the sentence is
# more related to the concept.
# In the approach of Gong and Liu, one sentence is chosen from the most important concept, and then a second sentence is chosen from the second most important concept until a predefined number of sentences are collected. The number of sentences to be collected is given as a parameter.
# In Example 1, three sentences were given, and the SVD calculations were performed accordingly. The resulting VT
# matrix having rank set to two is given in Figure 2. In this figure, first, the concept con0 is chosen, and then the sentence
# sent1 is chosen, since it has the highest cell value in that row.
# The approach of Gong and Liu has some disadvantages that are defined by Steinberger and Jezek [5]. The first disadvantage is that the number of sentences to be collected is the same with the reduced dimension. If the given predefined
# number is large, sentences from less significant concepts are chosen. The second disadvantage is related to choosing
# only one sentence from each concept. Some concepts, especially important ones, can contain sentences that are highly
# related to the concept, but do not have the highest cell value. The last disadvantage is that all chosen concepts are
# assumed to be in the same importance level, which may not be true.
def get_summary(sentenses, VT):
    log_size = np.log10(len(sentenses))
    max_items = int(log_size) if log_size >= 1.0 else 1
    summary_elements = []
    for k in range(max_items):
        values = VT[k].round(15)
        max_index = 0
        max_value = values[0]
        for i in range(1, len(values)):
            value = values[i]
            if value > max_value:
                max_value = value
                max_index = i
        summary_elements.append(max_index)

    summary_elements.sort()

    return "".join([sentenses[el] for el in summary_elements])


scores = np.array([])
rouges = np.array([])

sent_tokenizer = PunktSentenceTokenizer()
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'ms', 'u.s', 'rep'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

for i in range(len(document_list)):
    doc = document_list[i]
    terms = get_stemmed(doc)
    sentenses = sentence_splitter.tokenize(doc)
    if len(sentenses) == 0:
        continue
    idf_df, sentenses_stemmed = get_idf_dataframe(sentenses)
    term_doc_matrix = [[tfidf(term, sent, idf_df, sentenses_stemmed) for sent in sentenses] for term in terms]
    # 3.1.2. Step 2
    # Singular Value Decomposition: SVD is an algebraic method that can model relationships
    # among words/phrases and sentences. In this method, the given input matrix
    # A is decomposed into three new matrices as follows:
    U, S, VT = np.linalg.svd(term_doc_matrix)
    summary = get_summary(sentenses, VT)
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