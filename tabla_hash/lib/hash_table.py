""" 
Funcion principal
Integrantes: 
- Aguilar Wilson 
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""


def get_hash_table(data):
    b = len(data)
    table = [None] * b
    for x in data:
        position = get_position(x, b)
        hash(table, position, x)

    return table


def get_position(x, b):
    return x % b


def hash(table, pos, i):
    b = len(table)
    if table[pos] == None:
        table[pos] = i
    else:
        k = (i % (b - 1)) + 1
        h = (pos + k) % b
        hash(table, h, i)
