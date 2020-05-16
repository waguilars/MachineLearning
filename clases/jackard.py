


def get_jaccard(query, document):
    query = { i for i in query.split()}
    document = { i for i in document.split()}
    interseccion = len(query.intersection(document))
    union = len(query.union(document))
    return interseccion / union
    


res = get_jaccard('dias de lluvia', 'resbalo en un dia de lluvia')
res2 = get_jaccard('dias de lluvia', 'lluvia acida')
print(res)
print(res2)
