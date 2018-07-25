import matplotlib.pyplot as plt
import numpy as np

def inside(xi):
    # true if coord is inside ref cell
    TOL = 1e-8
    return xi[0] >=0.-TOL and xi[1] >= 0.-TOL and xi[0] + xi[1] <= 1.+TOL

phi =  [lambda xi: 1. - xi[0] - xi[1], \
        lambda xi: xi[0], \
        lambda xi: xi[1]]

if __name__ == '__main__':
    xi = np.linspace(0,1,11)
    yi = np.linspace(0,1,11)
    Xi, Yi = np.meshgrid(xi, yi)

    p = np.zeros((11,11))

    print phi[2]([1,0])

    # x is columns, y is rows
    for i in xrange(11):
        for j in xrange(11):
            x = [xi[i],yi[j]]
            if inside(x):
                p[j,i] = phi[1](x)

    plt.figure()
    #plt.contourf(Xi, Yi, p, 100)
    plt.contourf(xi,yi,p,100)
    plt.colorbar(shrink=0.8, extend='both')
    plt.xlabel('x')
    plt.show()

    print p



