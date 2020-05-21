import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv


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
        token_data = list(map(lambda x: get_positions(token, x), listadocs))
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
            #matches += [docs.index(doc), len(indexes), indexes]
            matches += [docs.index(doc), len(indexes)]
        if matches:
            all_matches.append(matches)

    if all_matches:
        return all_matches
    else:
        return None


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
    for item in fii:
        print(item, fii[item])
