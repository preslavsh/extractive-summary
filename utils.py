from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()


def load_data(count):
    documents_list = []
    summaries = []
    with open('1000.json') as file:
        string = file.read()
        entries = json.loads(string)
        i = 0
        for entry in entries:
            i = i + 1
            documents_list.append(entry["text"])
            summaries.append(entry["summary"])
            if i == count:
                break
    return documents_list, summaries


def get_stemmed(doc_to_stem):
    if doc_to_stem is not None:
        raw = doc_to_stem.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [a for a in tokens if not a in en_stop]
        return [p_stemmer.stem(j) for j in stopped_tokens]
    else:
        return []


def count_tokens(sent, topic):
    sum = 0
    stemmed_tokens = get_stemmed(sent)
    for token in stemmed_tokens:
        if token == topic:
            sum = sum + 1
    return sum


def calculate_vec(first_doc, second_doc):
    first_stem = get_stemmed(first_doc)
    second_stem = get_stemmed(second_doc)
    if len(first_stem) == 0 or len(second_stem) == 0:
        return 0.0
    total = set(first_stem).union(set(second_stem))
    word_dict_first = dict.fromkeys(total, 0)
    word_dict_second = dict.fromkeys(total, 0)
    for word in first_stem:
        if word in word_dict_first:
            word_dict_first[word] += 1

    for word in second_stem:
        if word in word_dict_second:
            word_dict_second[word] += 1
    a = list(word_dict_first.values())
    b = list(word_dict_second.values())
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return  cos_sim

def rougeL(candidate, reference):
    candidate_stem = get_stemmed(candidate)
    reference_stem = get_stemmed(reference)
    if len(candidate_stem) == 0 or len(reference_stem) == 0:
        return 0.0
    current_sequence = 0
    max_subsequence = 0
    for i in range(len(candidate_stem)):
        for j in range(len(reference_stem)):
            if j + i < len(candidate_stem) and (candidate_stem[j + i] == reference_stem[j]):
                current_sequence += 1
            else:
                if current_sequence > max_subsequence:
                    max_subsequence = current_sequence
                current_sequence = 0
    return max_subsequence/len(reference_stem)