from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# file = urlopen('https://dievalhu.github.io/diegovallejo/')
# html = file.read()
# file.close()

# soup = BeautifulSoup(html)
# # print(soup)

# searched = soup.find_all('a')

# tit = []
# for links in searched:
#     tit.append(links.get('href'))

# titles = tit[1:7]
# print(titles)

a = 'Researcher in Artificial Intelligence in the Department of Computer Science (IDEIAGEOCA Research Group) at the Universidad Politécnica Salesiana (UPS), Ecuador. Lecturer in the Department of Mathematics at the Universidad San Francisco de Quito (USFQ) and in the Department of Physics and Mathematics at the Universidad de las Américas (UDLA).'

# 1. normalizar
n1 = a.lower()
n2 = re.sub('[^A-Za-z0-9]', ' ', n1).split()

# print(n2)
print(len(n2))

# Eliminar stop words
n4 = stopwords.words('english')

# print(n4)

for word in n2:
    if word in n4:
        n2.remove(word)

# print(n2)
print(len(n2))

# steaming

stemmer = PorterStemmer()
n5 = stemmer.stem('Expierience')
# print(n5)
n6 = []

for i in n2:
    n6.append(stemmer.stem(i))

print(n6)
