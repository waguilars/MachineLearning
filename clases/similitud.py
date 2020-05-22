import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv
import numpy


def clean(a):
    b = a.lower()
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c


def to_string(titles):
    document = ""
    for item in titles:
        document += item+" "
    return document.strip()


def get_dict(cleaned_docs):
    data = []
    for doc in cleaned_docs:
        data += doc
    return list(set(data))


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


def new_fii(diccionario, listadocs):
    fii = {}
    for token in diccionario:
        token_data = list(
            map(lambda x: get_positions(token, x), listadocs))

        fii[token] = token_data
        # token_data = [token]
        # for doc in listadocs:
        #     positions = get_positions(token, doc)
        #     token_data.append(positions)
        # fii.append(token_data)
    return fii


def get_positions(token, docs):
    all_matches = []
    for doc in docs:
        matches = []
        if token in doc:
            indexes = [i for i, x in enumerate(doc) if x == token]
            # matches += [docs.index(doc), len(indexes), indexes]
            matches += [docs.index(doc), len(indexes)]
        if matches:
            all_matches.append(matches)

    if all_matches:
        return all_matches
    else:
        return None


def get_word_bag(fii, doc_list):

    frq_titles = []
    frq_keywords = []
    frq_abstract = []
    for word in fii:
        v_titles = fii[word][0]
        v_keywords = fii[word][1]
        v_abstract = fii[word][2]

        if v_titles is not None:
            frq_titles.append(sum(x[1] for x in v_titles))
        else:
            frq_titles.append(0)

        if v_keywords is not None:
            frq_keywords.append(sum(x[1] for x in v_keywords))
        else:
            frq_keywords.append(0)

        if v_abstract is not None:
            frq_abstract.append(sum(x[1] for x in v_abstract))
        else:
            frq_abstract.append(0)

    frq_titles = numpy.array(frq_titles)
    frq_keywords = numpy.array(frq_keywords)
    frq_abstract = numpy.array(frq_abstract)

    word_bag = numpy.column_stack((frq_titles, frq_keywords, frq_abstract))
    print(word_bag)


if __name__ == "__main__":
    data = pd.read_csv('/home/will/Descargas/data.csv', encoding='UTF-8')
    lista = []
    titles = list(data['title'])
    keywords = list(data['keywords'])
    abstract = list(data['abstract'])
    lista.append(titles)    # posicion 0
    lista.append(keywords)  # posicion 1
    lista.append(abstract)  # posicion 2
    diccionario = []

    for i in range(len(lista)):
        # limpieza
        lista[i] = list(map(clean, lista[i]))
        lista[i] = list(map(clean_stop_words, lista[i]))
        lista[i] = list(map(clean_stemmer, lista[i]))

        # obtener diccionario
        diccionario += get_dict(lista[i])

    diccionario = set(diccionario)

    fii = new_fii(diccionario, lista)

    get_word_bag(fii, lista)
