"""
Funcion principal
Integrantes:
- Aguilar Wilson
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""


def get_hash_table(data, size):
    b = size
    table = [None] * b
    for x in data:
        position = x % b
        if table[position] == None:
            table[position] = x
        else:
            k = (x % (b - 1)) + 1
            new_pos = re_disp(position, k, table)
            table[new_pos] = x
    return table


def re_disp(h, k, table):
    b = len(table)
    position = (h + k) % b

    if (table[position] == None):
        return position
    else:
        return re_disp(position, k, table)
