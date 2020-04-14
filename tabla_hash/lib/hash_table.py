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


# if __name__ == "__main__":
#     data = [23, 14, 9, 6, 30, 12, 15]
#     # data = [51, 14, 3, 7, 18, 30]
#     # sin segundo argumento intenta hacerlo demanera dinamica
#     hash_table = get_hash_table(data)
#     # hash_table = get_hash_table(data, 8) # con segundo parametro se fija el tamaño de la tabla hash
#     print(hash_table)
