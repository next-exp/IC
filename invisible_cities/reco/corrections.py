def Correction(xs, ys, E, _, __):
    N = E / E.max()
    norm = { (x,y) : N[i,j]
             for i, x in enumerate(xs)
             for j, y in enumerate(ys) }

    def correct(x, y):
        return norm[(x,y)]

    return correct
