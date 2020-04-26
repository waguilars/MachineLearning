from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def clean(a):
    """ Cambia a minusculas y limpia los caracteres
    especiales de un string

    params:
    a - string que se va a limpiar

    return:
    c - string en minusculas y limpio
    """
    b = a.lower()
    # en c se encuentra la lista de todos los documentos juntos y limpios
    c = re.sub('[^A-Za-z0-9]+', ' ', b).split()
    return c


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


tit2 = "".join(map(str, tit[17:66]))

f = clean(tit2)
print(len(f))
f2 = stopwords.words('english')
# print(f2)

for word in f:
    if word in f2:
        f.remove(word)

print(f)
# steemer=PorterStemmer()
# n5=steemer.stem('')
# n6=[]
# for i in f:
#     n6.append(steemer.stem(i))

# print(len(n6))
