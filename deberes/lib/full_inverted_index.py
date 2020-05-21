"""
Integrantes:
- Aguilar Wilson
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""
import re
import string


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


def get_dict(cleaned_docs):
    """ almacena en una variable, y por cada documento
        lo va aniadiendo a la lista y en caso de repetirse
        ya no la agrega

    params:
    cleaned_docs - documento realizado el proceso de normalizacion

    return:
    data - lista de palabras una sola vez
    """
    data = []
    for doc in cleaned_docs:
        data += doc
    return list(set(data))


def get_positions(token, docs):
    """ Obtiene el documento y las posiciones de
        un token dentro de un conjunto de documentos

        params:

        token - palabra a buscar

        docs - lista de documentos

        return:

        all_matches - lista con el token, numero de documento en la lista y las posiciones

    """

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

    newdoc = []
    for doc in docs:
        newdoc.append(clean(doc))
    docs = newdoc
    my_dict = get_dict(docs)
    fii = map(lambda x: get_positions(x, docs), my_dict)
    return list(fii)


if __name__ == "__main__":
    doc1 = "To do is to be. To be is to do hola hola"
    doc2 = "To be or not to be. I am what I am hola"
    doc3 = "I think therefore I am. Do be do be do. hola"
    doc4 = "Do do do, da da da. Let it be, let it be hola"

    docs = []
    docs.append(doc1)
    docs.append(doc2)
    docs.append(doc3)
    docs.append(doc4)

    fii = get_fii(docs)
    for item in fii:
        print(item)
