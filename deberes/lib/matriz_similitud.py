import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


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


def to_string(titles):
    document = ""
    for item in titles:
        document += item+" "
    return document.strip()

def do_nlp(lista):
    lista = list(map(clean, lista))
    lista = list(map(clean_stop_words, lista))
    lista = list(map(clean_stemmer, lista))
    lista = list(map(to_string, lista))
    return lista


def get_jackar(docs):
    labels = ['doc'+str(x) for x in range(len(docs))]
    jaccard_mtx = pd.DataFrame(float, index=labels, columns=labels)

    for i in range(len(docs)):
        for j in range(i, len(docs)):
            d1 = {i for i in docs[i].split()}
            d2 = {i for i in docs[j].split()}
            intersection = len(d1.intersection(d2))
            union = len(d1.union(d2))
            value = intersection/union

            jaccard_mtx._set_value('doc'+str(i), 'doc'+str(j), value)
            jaccard_mtx._set_value('doc'+str(j), 'doc'+str(i), value)
    return jaccard_mtx


if __name__ == "__main__":
    data = pd.read_csv('C:/Users/Ricardo/Desktop/dieguillo/python/ulti/dato.csv')
    titles = list(data['title'])
    keywords = list(data['keywords'])
    abstracts = list(data['abstract'])
    docs = []
    docs.append(titles)
    docs.append(keywords)
    docs.append(abstracts)
    docs = list(map(do_nlp, docs))

    titles = docs[0]
    keywords = docs[1]
    abstracts = docs[2]

    titles = get_jackar(titles)
    keywords = get_jackar(keywords)
    print(titles)
    print('----------------')
    print(keywords)

    # jackcard_titulos = get_jackar()  # ponderacion 10
    # jackcard_keywords = get_jackar(pd.DataFrame(
    #     float, index=nombres, columns=nombres), docs[1])  # ponderacion 30

