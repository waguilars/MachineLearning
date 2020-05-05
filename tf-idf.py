import math
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


def get_weight(tf):
    if tf and tf > 0:
        return 1+math.log10(tf)
    else:
        return 0


def clean(a):
    b = a.lower()
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c


def clean_stop_words(titles):
    stop_words = stopwords.words('english')
    for word in stop_words:
        if word in titles:
            titles.remove(word)
    return titles


def clean_stemmer(titles):
    stemmer = PorterStemmer()
    new_titles = []
    for item in titles:
        new_titles.append(stemmer.stem(item))
    return new_titles


def get_dict(cleaned_docs):
    data = []
    for doc in cleaned_docs:
        data += doc
    return list(set(data))


def get_positions(token, docs):
    all_matches = [token]
    values = []
    for doc in docs:
        if token in doc:
            indexes = [i for i, x in enumerate(doc) if x == token]
            # values.append([len(indexes), indexes])
            values.append(len(indexes))
        else:
            values.append(None)
    all_matches.append(values)
    return all_matches


def get_fii(docs):

    my_dict = get_dict(docs)
    fii = map(lambda x: get_positions(x, docs), my_dict)
    return list(fii)


""" 1 + log10 tf """
docs = []
docs.append('About Train, Validation and Test Sets in Machine Learning')
docs.append('This is aimed to be a short primer for anyone who needs to know the difference between the various dataset splits while training Machine Learning models.')
docs.append('For this article, I would quote the base definitions from Jason Brownlee’s excellent article on the same topic, it is quite comprehensive, if you like more details, do check it out.')
docs.append('The actual dataset that we use to train the model (weights and biases in the case of Neural Network). The model sees and learns from this data.')


docs = map(clean, docs)
docs = list(map(clean_stop_words, docs))
docs = list(map(clean_stemmer, docs))

my_dict = get_dict(docs)
fi = get_fii(docs)


for row in fi:
    row[1] = list(map(get_weight, row[1]))
    # for value in row[1]:
    #     new_values.append(get_weight(value))
    #     row[1] = new_values

#######


for item in fi:
    print(item)
