
def get_exponent_combos(degree):

    combos = list()
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j > degree:
                pass
            else:
                combos.append((i, j))

    return combos