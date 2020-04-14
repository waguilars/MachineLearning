"""
Funcion principal
Integrantes:
- Aguilar Wilson
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""


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
                print('el re-hash entro en bucle, intente con un tamaño de tabla mayor')
                exit()
    return table


def re_disp(h, k, table):
    b = len(table)
    position = (h + k) % b

    if (table[position] == None):
        return position
    else:
        return re_disp(position, k, table)
