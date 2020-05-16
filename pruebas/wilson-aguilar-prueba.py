from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

""" Wilson Aguilar """


""" Pregunta 1 """


def get_website_paragraphs(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    file = urlopen(req)
    html = file.read()
    file.close()
    soup = BeautifulSoup(html, 'html.parser')
    busca = soup.find_all("p")
    paragraphs = []
    for p in busca:
        paragraph = p.find_all(text=True, recursive=True)
        if paragraph != []:
            paragraphs.append(paragraph[0])
    new_paragraphs = []
    new_paragraphs.append(paragraphs[0])
    new_paragraphs.append(paragraphs[2])
    return new_paragraphs


""" Pregunta 2 """


def clean(a):
    b = a.lower()
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c


""" Pregunta 3 """


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


""" Pregunta 4 """


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
            matches += [docs.index(doc), len(indexes), indexes]
        if matches:
            all_matches.append(matches)
    return all_matches


def get_fii(docs):

    my_dict = get_dict(docs)
    fii = map(lambda x: get_positions(x, docs), my_dict)
    return list(fii)


""" Pregunta 5: y = 5x^2+3x+1. """


def encode_token(token):
    x = len(token)
    code = 5*(x**2)+3*x+1
    return code


def encode_fii(fii):
    encoded_fii = []
    for item in fii:
        aux = item
        aux[0] = encode_token(aux[0])
        encoded_fii.append(aux)
    return encoded_fii


"""  Pregunta 6 - hash"""


def get_hash_table(data, size=0):
    if size < len(data):
        b = len(data)
    else:
        b = size

    table = [None] * b
    for x in data:
        position = x % b
        if table[position] == None:
            table[position] = x
        else:
            k = (x % (b - 1)) + 1
            try:
                new_pos = re_disp(position, k, table)
                table[new_pos] = x
            except:
                print(
                    'No se pudo determinar la posicion del ultimo valor, entro en bucle.')
                return table

    return table


def re_disp(h, k, table):
    b = len(table)
    position = (h + k) % b

    if (table[position] == None):
        return position
    else:
        return re_disp(position, k, table)


""" Pregunta 6 """


def get_top_six(encoded_fii):
    ordered_list = []
    for item in encoded_fii:
        ordered_list.append(item[0])
    ordered_list = sorted(set(ordered_list), reverse=True)
    return ordered_list[0:6]


""" Pregunta 7 """


def check_tokens_in_hash_table(fii, hash_table):
    dictionary = []
    for item in fii:
        for value in hash_table:
            encoded = encode_token(item[0])
            if value and value == encoded:
                dictionary.append({value: item[0]})
    # data = []
    for value in hash_table:
        if value:
            print(value)
    print(dictionary[0])


if __name__ == "__main__":

    print('----------- Pregunta 1 -----------')
    paragraphs = get_website_paragraphs(
        'https://towardsdatascience.com/automated-text-classification-using-machine-learning-3df4f4f9570b')
    for item in paragraphs:
        print(item)

    print('----------- Pregunta 2 -----------')
    paragraphs = list(map(clean, paragraphs))
    for item in paragraphs:
        print(item)
    print('----------- Pregunta 3 -----------')
    paragraphs = list(map(clean_stop_words, paragraphs))
    paragraphs = list(map(clean_stemmer, paragraphs))
    for item in paragraphs:
        print(item)

    print('----------- Pregunta 4 - Full inverted index -----------')
    my_dict = get_dict(paragraphs)
    fii = get_fii(paragraphs)

    for item in fii:
        print(item)

    print('------------ pregunta 5 - Encoded full inverted index ----------------')

    new_fii = encode_fii(fii)
    for item in new_fii:
        print(item)

    print('------------ pregunta 6 - Hash table ----------------')
    top = get_top_six(new_fii)
    hash_table = get_hash_table(top, 10)
    print(hash_table)
    print('------------ pregunta 7 - pablas en tabla hash ----------------')
    fii = get_fii(paragraphs)
    check_tokens_in_hash_table(fii, hash_table)
