import numpy as np
from numpy.linalg import inv

def get_psi(m):
    """ get prime basis function
     m - exponent for xi """
    if len(m) == 2:
        def psi(xi):
            return (xi[0]**m[0])*(xi[1]**m[1])
    else: # dim = 1
        def psi(xi):
            return xi[0]**m[0]
    return psi

def get_dpsi(m):
    """ get prime basis function derivative
     m - exponent for xi """
    if len(m) == 2: # dim = 2
        def dpsi(xi):
            if m[0] == 0:
                comp1 = 0
            else:
                comp1 = m[0]*(xi[0]**(m[0]-1))*(xi[1]**m[1])

            if m[1] == 0:
                comp2 = 0
            else:
                comp2 = m[1]*(xi[1]**(m[1]-1))*(xi[0]**m[0])

            return np.array([comp1, comp2])
    else: # dim = 1
        def dpsi(xi):
            if m[0] == 0:
                return 0
            else:
                return m[0]*xi[0]**(m[0]-1)
    return dpsi

def prime_basis(deg, dim):
    """ get complete prime basis and derivatives
    deg - degree polynomial space
    dim - spatial dimension """
    if dim == 2:
        psi = [get_psi([m,n]) for m in xrange(deg+1) for n in xrange(deg+1) if m+n<=deg]
        dpsi = [get_dpsi([m,n]) for m in xrange(deg+1) for n in xrange(deg+1) if m+n<=deg]
    else: # dim = 1
        psi = [get_psi([m]) for m in xrange(deg+1) if m<=deg]
        dpsi = [get_dpsi([m]) for m in xrange(deg+1) if m<=deg]
    return psi, dpsi

def get_nodes(deg, dim):
    """ get nodal points in ref elt
    deg - degree of polynomial space (number of nodes)
    dim - spatial dimension (triangle or interval) """
    if dim == 2:
        #pts = np.linspace(0,1,deg+1)
        #return [[xi,yi] for yi in pts for xi in pts if xi+yi<=1]
        if deg == 1:
            nodes = [[0.,0.], [1.,0.], [0.,1.]]
        elif deg == 2:
            nodes = [[0.,0.], [1.,0.], [0.,1.], [.5,0.], [.5,.5], [0.,.5]]
        elif deg == 3:
            nodes = [[0.,0.], [1.,0.], [0.,1.], [1./3,0.], [2./3,0.],\
                     [2./3, 1./3], [1./3, 2./3], [0.,1./3], [0.,2./3], [1./3, 1./3]]
        return nodes
    else: # dim = 1
        nodes = [[xi] for xi in np.linspace(-1,1,deg+1)]
        return [nodes[0], nodes[-1]] + [nodes[i] for i in range(1,deg)]

def get_phi(i, coeffs, psi, n_nodes):
    """ get i'th Lagrange basis function or derivative
    coeffs - coefficient matrix for expansion in prime basis
    psi - complete prime basis or derivative
    n_nodes - number of nodes """
    def phi(xi):
        return np.sum([coeffs[i,j]*psi[j](xi) for j in xrange(n_nodes)], axis=0)
    return phi

def get_lagrange_basis(deg, dim):
    """ get complete Lagrange basis
    deg - degree of polynomial space
    dim - spatial dimension """
    nodes = get_nodes(deg,dim); n_nodes = len(nodes)
    psi, dpsi = prime_basis(deg, dim)

    # assmeble coefficient matrix
    V = np.zeros((n_nodes, n_nodes))
    for i in xrange(n_nodes):
        for j in xrange(n_nodes):
            V[i,j] = psi[j](nodes[i])
    coeffs = inv(V).T

    phi = [get_phi(i, coeffs, psi, n_nodes) for i in xrange(n_nodes)]
    dphi = [get_phi(i, coeffs, dpsi, n_nodes) for i in xrange(n_nodes)]

    return phi, dphi

def get_rt_basis():
    """ get lowest order Raviart Thomas basis functions (only 2D) """
    phi = [lambda x: [x[0], x[1]],\
           lambda x: [x[0]-1., x[1]],\
           lambda x: [x[0], x[1]-1.]]

    dphi = [lambda x: 2.,\
            lambda x: 1.,\
            lambda x: 1.]

    return phi, dphi


class ShapeFunctions:
    """  Shape functions defined on reference element """
    def __init__(self, method, deg, dim):
        """
        method - 'lagrange' or 'RT'
        deg - degree of shape functions (0,1,2,3 for lagrange, 0 for RT)
        dim - spatial dimension (1,2)
        """
        if method == 'lagrange':
            # check valid dimentions and degree
            assert dim == 1 or dim == 2, 'Invalid dimension for Lagrange shape functions.'
            assert isinstance(deg, int) and (0 <= deg <= 3), 'Invalid degree for Lagrange shape functions.'

            # get shape functions
            if deg != 0:
                phi, dphi = get_lagrange_basis(deg, dim)
            else:
                phi, dphi = [None,None]
        elif method == 'rt':
            assert dim == 2, 'Invalid dimension for RT shape functions.'
            assert deg == 0, 'Invalid degree for RT shape functions'

            # get shape functions
            phi, dphi = get_rt_basis()
        else:
            raise ValueError('Invalid method, no shape functions available.')

        self.__phi = phi
        self.__dphi = dphi
        self.__method = method
        #self.__deg = deg # <-------- need this?


    def eval(self, n, xi, derivative=False):
        """ eval n'th shape function, either phi or dphi, at xi """
        if self.__phi is None:
            assert not derivative, 'No derivative of P0 shape functions.'
            return 1.
        else:
            if derivative:
                return self.__dphi[n](xi)
            else:
                return self.__phi[n](xi)

    def n_dofs(self):
        """ return number of DOFs """
        if self.__phi is None:
            return 1
        else:
            return len(self.__phi)

    def method(self):
        """ return type of shape function """
        return self.__method




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def inside(xi):
        """ true if xi is inside ref cell """
        TOL = 1e-8
        if len(xi) == 2:
            return xi[0] >=0.-TOL and xi[1] >= 0.-TOL and xi[0] + xi[1] <= 1.+TOL
        elif len(xi) == 1:
            return xi[0] >= -1. and xi[0] <= 1.

    def plot1D():
        """ plot 1D Lagrange shape functions on ref element """
        xi = np.linspace(-1,1,101)

        p1 = ShapeFunctions('lagrange',1,1)
        p2 = ShapeFunctions('lagrange',2,1)
        p3 = ShapeFunctions('lagrange',3,1)

        p1_ = np.zeros((101,2)); p2_ = np.zeros((101,3)); p3_ = np.zeros((101,4))
        for n in xrange(101):
            p1_[n,0] = p1.eval(0,[xi[n]])
            p1_[n,1] = p1.eval(1,[xi[n]])

            p2_[n,0] = p2.eval(0,[xi[n]])
            p2_[n,1] = p2.eval(1,[xi[n]])
            p2_[n,2] = p2.eval(2,[xi[n]])

            p3_[n,0] = p3.eval(0,[xi[n]])
            p3_[n,1] = p3.eval(1,[xi[n]])
            p3_[n,2] = p3.eval(2,[xi[n]])
            p3_[n,3] = p3.eval(3,[xi[n]])

        plt.figure(num='1D Lagrange shape functions', figsize=(12,6))

        plt.subplot(131)
        plt.plot(xi, p1_[:,0])
        plt.plot(xi, p1_[:,1])
        plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)
        plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)
        plt.title('Linear')
        plt.xlim([-1,1])
        plt.ylim([-.35,1.1])

        plt.subplot(132)
        plt.plot(xi, p2_[:,0])
        plt.plot(xi, p2_[:,1])
        plt.plot(xi, p2_[:,2])
        plt.axvline(x=0., color='k', linestyle='--', linewidth=.5)
        plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)
        plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)
        plt.title('Quadratic')
        plt.xlim([-1,1])
        plt.ylim([-.35,1.1])

        plt.subplot(133)
        plt.plot(xi, p3_[:,0])
        plt.plot(xi, p3_[:,1])
        plt.plot(xi, p3_[:,2])
        plt.plot(xi, p3_[:,3])
        plt.axvline(x=-1./3, color='k', linestyle='--', linewidth=.5)
        plt.axvline(x=1./3, color='k', linestyle='--', linewidth=.5)
        plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)
        plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)
        plt.title('Cubic')
        plt.xlim([-1,1])
        plt.ylim([-.35,1.1])

        plt.show()

    def plot2D():
        """ plot 2D Lagrange shape functions on ref element """
        xi = np.linspace(0,1,101)
        yi = np.linspace(0,1,101)
        #Xi, Yi = np.meshgrid(xi, yi)

        p1 = ShapeFunctions('lagrange',1,2)
        p2 = ShapeFunctions('lagrange',2,2)
        p3 = ShapeFunctions('lagrange',3,2)

        p11 = np.zeros((101,101))
        p12 = np.zeros((101,101))
        p13 = np.zeros((101,101))

        p21 = np.zeros((101,101))
        p22 = np.zeros((101,101))
        p23 = np.zeros((101,101))
        p24 = np.zeros((101,101))
        p25 = np.zeros((101,101))
        p26 = np.zeros((101,101))

        p31 = np.zeros((101,101))
        p32 = np.zeros((101,101))
        p33 = np.zeros((101,101))
        p34 = np.zeros((101,101))
        p35 = np.zeros((101,101))
        p36 = np.zeros((101,101))
        p37 = np.zeros((101,101))
        p38 = np.zeros((101,101))
        p39 = np.zeros((101,101))
        p310 = np.zeros((101,101))


        # x is columns, y is rows
        for i in xrange(101):
            for j in xrange(101):
                x = [xi[i],yi[j]]
                if inside(x):
                    p11[j,i] = p1.eval(0,x)
                    p12[j,i] = p1.eval(1,x)
                    p13[j,i] = p1.eval(2,x)

                    p21[j,i] = p2.eval(0,x)
                    p22[j,i] = p2.eval(1,x)
                    p23[j,i] = p2.eval(2,x)
                    p24[j,i] = p2.eval(3,x)
                    p25[j,i] = p2.eval(4,x)
                    p26[j,i] = p2.eval(5,x)

                    p31[j,i] = p3.eval(0,x)
                    p32[j,i] = p3.eval(1,x)
                    p33[j,i] = p3.eval(2,x)
                    p34[j,i] = p3.eval(3,x)
                    p35[j,i] = p3.eval(4,x)
                    p36[j,i] = p3.eval(5,x)
                    p37[j,i] = p3.eval(6,x)
                    p38[j,i] = p3.eval(7,x)
                    p39[j,i] = p3.eval(8,x)
                    p310[j,i] = p3.eval(9,x)

        # plot linear functions
        plt.figure(num='2D linear shape functions', figsize=(6,7))
        plt.subplot(311)
        CS1 = plt.contourf(xi, yi, p11, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(312)
        CS1 = plt.contourf(xi, yi, p12, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(313)
        CS1 = plt.contourf(xi, yi, p13, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_3')
        plt.axis('off')

        # plot quadratic functions
        plt.figure(num='2D quadratic shape functions', figsize=(10,7.5))
        plt.subplot(321)
        CS1 = plt.contourf(xi, yi, p21, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(322)
        CS1 = plt.contourf(xi, yi, p22, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(323)
        CS1 = plt.contourf(xi, yi, p23, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_3')
        plt.axis('off')

        plt.subplot(324)
        CS1 = plt.contourf(xi, yi, p24, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_4')
        plt.axis('off')

        plt.subplot(325)
        CS1 = plt.contourf(xi, yi, p25, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_5')
        plt.axis('off')

        plt.subplot(326)
        CS1 = plt.contourf(xi, yi, p26, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_6')
        plt.axis('off')

        # plot cubic functions
        plt.figure(num='2D cubic shape functions', figsize=(14,7.5))
        plt.subplot(251)
        CS1 = plt.contourf(xi, yi, p31, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(252)
        CS1 = plt.contourf(xi, yi, p32, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(253)
        CS1 = plt.contourf(xi, yi, p33, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_3')
        plt.axis('off')

        plt.subplot(254)
        CS1 = plt.contourf(xi, yi, p34, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_4')
        plt.axis('off')

        plt.subplot(255)
        CS1 = plt.contourf(xi, yi, p35, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_5')
        plt.axis('off')

        plt.subplot(256)
        CS1 = plt.contourf(xi, yi, p36, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_6')
        plt.axis('off')

        plt.subplot(257)
        CS1 = plt.contourf(xi, yi, p37, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_7')
        plt.axis('off')

        plt.subplot(258)
        CS1 = plt.contourf(xi, yi, p38, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_8')
        plt.axis('off')

        plt.subplot(259)
        CS1 = plt.contourf(xi, yi, p39, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_9')
        plt.axis('off')

        plt.subplot(2, 5, 10)
        CS1 = plt.contourf(xi, yi, p310, 100)
        CB = plt.colorbar(CS1, shrink=0.8, extend='both')
        plt.title('phi_10')
        plt.axis('off')

        plt.show()

    #plot1D()
    #plot2D()







