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
        position = hash(x, b)
        if table[position] == None:
            table[position] = x
            print(table)
        else:
            print('redispersar ', x, position)
            # TODO: hacer funcion para redispersar
            return


def hash(x, b):
    return x % b

# TODO: Funcion incompleta


def redisp(x, data):
    b = len(data)
    kx = (x % (b-1)) + 1
