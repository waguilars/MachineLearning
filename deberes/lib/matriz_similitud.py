import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math as ma


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


def get_dict(cleaned_docs):
    data = []
    for doc in cleaned_docs:
        data += doc
    return list(set(data))


def get_positions(token, docs):
    all_matches = [token]
    for doc in docs:
        matches = []
        if token in doc:
            indexes = [i for i, x in enumerate(doc) if x == token]
            # matches += [docs.index(doc), len(indexes), indexes]
            matches += [docs.index(doc), len(indexes)]
        if matches:
            all_matches.append(matches)
    return all_matches


def get_fii(docs):

    newdoc = []
    for doc in docs:
        newdoc.append(clean(doc))
    docs = newdoc
    my_dict = get_dict(docs)
    fii = map(lambda x: get_positions(x, docs), my_dict)
    return list(fii)


def get_tf_word_bag(fii, palabras, documentos, weighted=True):
    tb_tf = pd.DataFrame(float(0), index=palabras, columns=[
                         'doc'+str(x) for x in range(len(abstracts))])
    for i in fii:
        con = 0
        for j in i:
            if con != 0:
                if weighted == True:
                    tb_tf._set_value(i[0], "doc"+str(j[0]), j[1])  # tabla tf
                else:
                    tb_tf._set_value(i[0], "doc"+str(j[0]),
                                     (1+ma.log(j[1], 10)))  # tabla wtf
            con += 1
    return tb_tf


def get_df_idf(palabras, tb_tf, tb_wtf, idf=True):
    df = pd.DataFrame(float(0), index=palabras, columns=['frecuency'])
    for index, row in tb_tf.iterrows():
        con = 0
        for i, ind in row.iteritems():
            if ind != 0:
                con += 1
        if idf == True:
            if con != 0:
                op = ma.log((len(tb_wtf.columns)/con), 10)
                df._set_value(index, 'frecuency', op)
            else:
                df._set_value(index, 'frecuency', con)
        else:
            df._set_value(index, 'frecuency', con)
    return df


def get_mtx_tf_idf(palabras, abstracts, tb_wtf, idf):
    tb_tf_idf = pd.DataFrame(float(0), index=palabras, columns=[
                             'doc'+str(x) for x in range(len(abstracts))])
    for index, row in tb_wtf.iterrows():
        for i, ind in row.iteritems():
            # index nombre fila , # i columna nombre, #ind term frecuency
            tb_tf_idf._set_value(
                index, i, (ind*idf._get_value(index, 'frecuency')))
    return tb_tf_idf


def normalize_tf_idf(tf_idf):
    tf = tf_idf
    nom = tf.columns.tolist()
    for i in nom:
        columna = tf[i].tolist()
        res = ma.sqrt(sum(value**2 for value in columna))
        div = [value/res for value in columna]
        tf[i] = div
    return tf


def get_cos_mtx(tf_idf_mtx):
    labels = tf_idf_mtx.columns
    cos_mtx = pd.DataFrame(float, index=labels, columns=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):

            doc1 = tf_idf_mtx['doc'+str(i)].tolist()
            doc2 = tf_idf_mtx['doc'+str(j)].tolist()
            value = sum(val1*val2 for val1, val2 in zip(doc1, doc2))
            cos_mtx['doc'+str(i)]['doc'+str(j)] = value

    return cos_mtx


if __name__ == "__main__":
    data = pd.read_csv('/home/will/Descargas/data.csv')
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
    # print(titles)
    # print('----------------')
    # print(keywords)
    fii = get_fii(abstracts)
    palabras = []
    for i in fii:
        palabras.append(i[0])
    tf = get_tf_word_bag(fii, palabras, abstracts, True)  # term frecuency
    wtf = get_tf_word_bag(fii, palabras, abstracts,
                          False)  # weight term frecuency
    # print(tf)
    # print(wtf)

    df = get_df_idf(palabras, tf, wtf, False)  # document frecuency
    idf = get_df_idf(palabras, tf, wtf, True)  # invert document frecuency
    # print(df)
    # print(idf)

    tb_tf_idf = get_mtx_tf_idf(palabras, abstracts, wtf, idf)
    print(tb_tf_idf)
    n_tf_idf = normalize_tf_idf(tb_tf_idf)
    print(n_tf_idf)
    print('-------------------')

    sim_cos = get_cos_mtx(n_tf_idf)
    print(sim_cos)
