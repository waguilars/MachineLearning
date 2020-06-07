######################################
#######       Wilson Aguilar  ########
######################################
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv
import pandas as pd
import math as ma
import nltk
nltk.download('stopwords')


def import_data(url):
    data = pd.read_csv(url)
    return data


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
    my_dict = get_dict(docs)
    fii = map(lambda x: get_positions(x, docs), my_dict)
    return list(fii)


def get_binary_mtx(dictionary, docs):
    mtx = pd.DataFrame(int(0), index=dictionary, columns=[
                       i for i in range(len(docs))])

    for word in dictionary:
        for doc in docs:
            if word in doc:
                index = int(docs.index(doc))
                mtx[index][word] = 1
    return mtx


def get_tf_word_bag(fii, palabras, documentos, weighted=True):
    tb_tf = pd.DataFrame(float(0), index=palabras, columns=[
                         'doc'+str(x) for x in range(len(abstracts))])
    for i in fii:
        con = 0
        for j in i:
            if con != 0:
                if weighted == False:
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
        for j in range(i, len(labels)):

            doc1 = tf_idf_mtx['doc'+str(i)].tolist()
            doc2 = tf_idf_mtx['doc'+str(j)].tolist()
            value = sum(val1*val2 for val1, val2 in zip(doc1, doc2))
            cos_mtx['doc'+str(i)]['doc'+str(j)] = value
            cos_mtx['doc'+str(j)]['doc'+str(i)] = value

    return cos_mtx


def get_jaccard(docs):
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
    csv_data = import_data(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00314/%5bUCI%5d%20AAAI-13%20Accepted%20Papers%20-%20Papers.csv')  # desde la web
    # csv_data = import_data('dataset.csv') # archivo local
    print('--------------- Pregunta 1 - importar dataset')
    print(csv_data)

    print('--------------- Pregunta 3 - NLP normalizacion 5 primeros abstract ------------------')
    abstracts = csv_data['Abstract'][0:5].tolist()
    abstracts = list(map(clean, abstracts))
    cleaned_abstr = list(map(to_string, abstracts))
    print(cleaned_abstr)

    print('--------------- Pregunta 4 - stopwords steming tokenizacion ------------------')
    abstracts = list(map(clean_stop_words, abstracts))
    abstracts = list(map(clean_stemmer, abstracts))
    print(abstracts)

    print('--------------- Pregunta 5 - Full inverted index------------------')
    fii = get_fii(abstracts)
    for word in fii:
        print(word)

    print('--------------- Pregunta 6 - Matriz de incidencia binaria------------------')
    my_dict = get_dict(abstracts)
    binary_mtx = get_binary_mtx(my_dict, abstracts)
    print(binary_mtx)
    print('Dimenciones de la matriz: ', binary_mtx.shape)

    print('--------------- Pregunta 6 - Matriz de TF_IDF------------------')
    tf = get_tf_word_bag(fii, my_dict, abstracts, False)
    wtf = get_tf_word_bag(fii, my_dict, abstracts, True)
    df = get_df_idf(my_dict, tf, wtf, False)
    idf = get_df_idf(my_dict, tf, wtf, True)

    tf_idf = get_mtx_tf_idf(my_dict, abstracts, wtf, idf)

    print(tf_idf)
    print('Dimensiones: ', tf_idf.shape)

    print('--------------- Pregunta 7 - Jaccard similitud D3 y D4------------------')
    cleaned_abstr = list(map(to_string, abstracts))
    jaccard_mtx = get_jaccard(cleaned_abstr)
    print(jaccard_mtx)  # D3 = doc2 y D4 = doc3
    D3 = 'doc2'
    D4 = 'doc3'
    print('Similitud entre D3 y D4 es: ', round(
        jaccard_mtx[D3][D4] * 100, 2), '%')
