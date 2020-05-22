
#     d1 d2    d3
#  d1  1  0.284
#  d2
#  d3
#
#
#
#

def get_jaccard(query, document):
    query = {i for i in query.split()}
    document = {i for i in document.split()}
    interseccion = len(query.intersection(document))
    union = len(query.union(document))
    return interseccion / union


res = get_jaccard('dias de lluvia', 'resbalo en un dia de lluvia')
res2 = get_jaccard('dias de lluvia', 'dias de lluvia')

print(res)
print(res2)

a = ['hola', 'mundo']
b = set(a)
print(a)
print(b)

print([i for i in range(2, 5)])
