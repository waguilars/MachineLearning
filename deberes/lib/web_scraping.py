"""
Integrantes:
- Aguilar Wilson
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv

<<<<<<< HEAD
=======

def clean(a):
    """ Cambia a minusculas y limpia los caracteres
    especiales de un string

    params:
    a - string que se va a limpiar
>>>>>>> 75aa490202a6c020588cec35d61cd9531c5bde7f

def clean(a):
    b = a.lower()
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c

def export_page_titles(url):
    file = urlopen(url)
    html = file.read()
    file.close()
    soup = BeautifulSoup(html, 'html.parser')
    busca = soup.find_all("a")
    titles = []
    for links in busca:
        title = links.find_all(text=True, recursive=False)
        if title != []:
            titles.append(title[0])
    titles = titles[11:23]
    # export to csvt
    with open('page_titles.csv', mode='w') as my_file:
        file_writer = csv.writer(
            my_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for title in titles:
            my_file.write(title+'\n')


def import_page_titles():
    with open('page_titles.csv', newline='') as File:
        reader = csv.reader(File)
        titles = []
        for row in reader:
            titles.append(row[0])
        return titles


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
            matches += [docs.index(doc), len(indexes), indexes]
        if matches:
            all_matches.append(matches)
    return all_matches


<<<<<<< HEAD
def get_fii(titles):
    newdoc = []
    for doc in titles:
        newdoc.append(clean(doc))
    my_dict = get_dict(newdoc)
    fii = map(lambda x: get_positions(x, newdoc), my_dict)
    return list(fii)

=======
file = urlopen('https://archive.ics.uci.edu/ml/index.php')
html = file.read()
file.close()

soup = BeautifulSoup(html)
busca = soup.find_all("a")
tit = []
con = 0
for links in soup.find_all('a'):
    # con+=1
    # print(con,links.get('href'))
    tit.append(links.get('href'))
>>>>>>> 75aa490202a6c020588cec35d61cd9531c5bde7f

def to_string(titles):
    document = ""
    for item in titles:
        document += item+" "
    return document.strip()

<<<<<<< HEAD

def web_scraping(url_page):
    export_page_titles(url_page)
    titles = import_page_titles()
    titles = map(clean, titles)
    titles = list(map(clean_stop_words, titles))
    titles = list(map(clean_stemmer, titles))
    titles = list(map(to_string, titles))
    fii = get_fii(titles)
    return fii
=======
tit2 = "".join(map(str, tit[17:66]))

f = clean(tit2)
print(len(f))
f2 = stopwords.words('english')
# print(f2)
>>>>>>> 75aa490202a6c020588cec35d61cd9531c5bde7f


<<<<<<< HEAD
if __name__ == "__main__":
    url_page = 'https://archive.ics.uci.edu/ml/index.php'
    ws_fii = web_scraping(url_page)
    for item in ws_fii:
        print(item)
=======
print(f)
# steemer=PorterStemmer()
# n5=steemer.stem('')
# n6=[]
# for i in f:
#     n6.append(steemer.stem(i))

# print(len(n6))
>>>>>>> 75aa490202a6c020588cec35d61cd9531c5bde7f
