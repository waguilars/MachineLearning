""" 
Integrantes: 
- Aguilar Wilson 
- Cacuango Gabriel
- Lasso Christian
- Romo Ricardo
"""


def get_code(token):
    k = len(token) - 1
    a = 2
    index = 0
    code = 0
    while index < len(token):
        character = token[index]
        code += ord(character) * (a**k)
        k -= 1
        index += 1
    return code


if __name__ == "__main__":
    my_token = "hola"
    polinomial = get_code(my_token)
    print(" polinomial= ", polinomial)

