import re
import timeit
import math as ma
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy
import nltk
import math as m


def clean(a):
    b = a.lower()
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c


def to_string(titles):
    document = ""
    for item in titles:
        document += item+" "
    return document.strip()


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


def get_word_bag(fii):
    qv = {}
    dv = {}
    for item in fii:
        # qv[item] = fii[item][0]
        # dv[item] = fii[item][1]
        val_q = fii[item][0]
        if val_q:
            tf = sum(x[1] for x in val_q)
        else:
            tf = 0

        qv[item] = tf

        val_d = fii[item][1]
        if val_d:
            tf = sum(x[1] for x in val_d)
        else:
            tf = 0

        dv[item] = tf
    labels = list(fii.keys())
    col_names = ['queries', 'documents']
    mtx = pd.DataFrame(float, index=labels, columns=col_names)
    for item in labels:
        mtx['queries'][item] = qv[item]
        mtx['documents'][item] = dv[item]

    return mtx


def get_wtf(tf):
    if tf and tf > 0:
        return 1+m.log10(tf)
    else:
        return 0


def get_wtf_mtx(mtx):
    new_mtx = mtx.copy()
    wqueries = new_mtx['queries'].tolist()
    wqueries = [get_wtf(x) for x in wqueries]
    new_mtx['queries'] = wqueries

    wdocs = new_mtx['documents'].tolist()
    wdocs = [get_wtf(x) for x in wdocs]
    new_mtx['documents'] = wdocs

    return new_mtx


def get_df(fii, index, docs):
    # 0 query
    # 1 doc
    dfs = {}
    for key in fii:
        values = fii[key][index]

        if values:
            dfs[key] = len(values)
            dfs[key] = m.log10(len(docs) / dfs[key])
        else:
            dfs[key] = 0
    return dfs


def get_tf_idf_mtx(wtf):
    new_mtx = wtf.copy()
    q = get_df(fii, 0, queries)
    d = get_df(fii, 1, documents)

    for key in q:
        new_mtx['queries'][key] = d[key] * new_mtx['queries'][key]
        new_mtx['documents'][key] = d[key] * new_mtx['documents'][key]

    return new_mtx


if __name__ == "__main__":
    Q1 = "“Machine learning with small Datasets”"
    Q2 = "“Probabilisitic model in cancer experiments”"
    Q3 = "“Learning in tasks with small data and Classifications models”"

    D1 = "“Transfer learning considers related but distinct tasks defined on heterogenous domains and tries to transfer knowledge between these tasks to improve generalization performance. It is particularly useful when we do not have sufficient amount of labeled training data in some tasks, which may be very costly, laborious, or even infeasible to obtain. Instead, learning the tasks jointly enables us to effectively increase the amount of labeled training data. In this paper, we formulate a kernelized Bayesian transfer learning framework that is a principled combination of kernel-based dimensionality reduction models with task-specific projection matrices to find a shared subspace and a coupled classification model for all of the tasks in this subspace.”"

    D2 = "“Our two main contributions are: (i) two novel probabilistic models for binary and multiclass classification, and (ii) very efficient variational approximation procedures for these models. We illustrate the generalization performance of our algorithms on two different applications. In computer vision experiments, our method outperforms the state-of-the-art algorithms on nine out of 12 benchmark supervised domain adaptation experiments defined on two object recognition data sets.”"

    D3 = "“In cancer biology experiments, we use our algorithm to predict mutation status of important cancer genes from gene expression profiles using two distinct cancer populations, namely, patient-derived primary tumor data and in-vitro-derived cancer cell line data. We show that we can increase our generalization performance on primary tumors using cell lines as an auxiliary data source.”"

    queries = [Q1, Q2, Q3]
    documents = [D1, D2, D3]

    print('---------------- Pregunta 1 - limpieza----------------------')
    queries = list(map(clean, queries))
    queries_cleaned = list(map(to_string, queries))
    documents = list(map(clean, documents))
    documents_cleaned = list(map(to_string, documents))
    print('queries:\n', queries)
    print('documents:\n', documents)

    print('---------------- Pregunta 2 - stop steemer token----------------------')
    queries = list(map(clean_stop_words, queries))
    documents = list(map(clean_stop_words, documents))
    queries = list(map(clean_stemmer, queries))
    documents = list(map(clean_stemmer, documents))

    print('queries:\n', queries)
    print('documents:\n', documents)

    print('---------------- Pregunta 3 - word bag----------------------')

    my_dict_queries = get_dict(queries)
    my_dict_documents = get_dict(documents)
    my_dict = set(my_dict_queries + my_dict_documents)

    fii = new_fii(my_dict, [queries, documents])
    wb = get_word_bag(fii)

    print(wb)
    print('La dimension es: ', wb.shape)

    print('---------------- Pregunta 4 - pesado tf----------------------')
    wtf = get_wtf_mtx(wb)
    print(wtf)

    print('---------------- Pregunta 5 - idf queries----------------------')
    # get_idf_row(wb)
    df_queries = get_df(fii, 0, queries)
    for key in df_queries:
        print(key, df_queries[key])

    print('---------------- Pregunta 6 - tf- idf ----------------------')
    tf_idf = get_tf_idf_mtx(wtf)
    print(tf_idf)
