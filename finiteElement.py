import numpy as np
from numpy.linalg import norm
from gaussquad import gaussPts
from shapefuncs import ShapeFunctions

def affine_map(vertices):
    """ affine transformation F : K_ref -> K, where F(x) = a x + b, for x in K_ref
    Input:
        vertices - local vertices to coordinate matrix
    Output:
        B, b, det (2D: meas = |det|/2, 1D: meas = |det|)
    """
    if len(vertices) == 3:
        tmp1 = vertices[1] - vertices[0]
        tmp2 = vertices[2] - vertices[0]

        a = np.array([tmp1, tmp2]).T
        a_inv = np.linalg.inv(a)
        det = np.linalg.det(a)
        b = vertices[0]
    else:
        tmp1 = (vertices[1] - vertices[0])/2
        tmp2 = 2*np.ones(2)

        a = np.array([tmp1, tmp2]).T
        #a_inv = np.linalg.inv(a)
        a_inv = None
        det = np.linalg.det(a)
        b = (vertices[0] + vertices[1])/2
    return a, a_inv, det, b

#--------------------------------------------------------------------------------------#

def check_method(method):
    """ checks methods are valid """
    method = map(lambda x: x.lower(), method)
    for m in method:
        assert any(map(lambda x: x == m, ['lagrange', 'rt'])), 'Method number {} is invalid.'.format(m)
    return method

#--------------------------------------------------------------------------------------#

class FiniteElement:
    """ Finite Element class """
    def __init__(self, method='lagrange', deg=1):
        """ Input:
            method - finite element method
            deg - degree of finite element method
        """
        if isinstance(method, list):
            assert len(method) == len(deg), 'Number of degrees and methods must be equal.'
            self.__mixed = True
        else:
            method = [method]; deg = [deg]
            self.__mixed = False

        self.__method = check_method(method)
        self.__n_methods = len(method)
        self.__deg = deg

    def mixed(self):
        """ True if mixed finite element """
        return self.__mixed

    def method(self, m=None):
        """ returns finite element method """
        if m is None:
            return self.__method
        else:
            return self.__method[m]

    def deg(self, m=None):
        """ returns degree """
        if m is None:
            return self.__deg
        else:
            return self.__deg[m]

    def initialize(self, dim):
        """ initialize element
        dim - spatial dimension of finite element """

        # set shape functions
        sfns = []; n_dofs = []
        for n in xrange(self.__n_methods):
            sf = ShapeFunctions(self.__method[n], self.__deg[n], dim)
            n_dofs.append(sf.n_dofs())
            sfns.append(sf)

        self.__dim = dim
        self.__sfns = sfns
        self.__n_dofs = n_dofs

    def n_dofs(self, m=0):
        """ returns number of dofs """
        return self.__n_dofs[m]

    def set_vertices(self, vertices):
        """ set local vertex to coordinate matrix """
        assert len(vertices)-1 == self.__dim, 'Number of vertices and dimension mismatch.'

        # initialize affine transform
        a, a_inv, det, b = affine_map(vertices)
        self.__a = a
        self.__a_inv = a_inv
        self.__det = det
        self.__b = b

        # set vertices
        self.__vertices = vertices

    def measure(self):
        """ returns measure of element (area in 2d, length in 1d) """
        return abs(self.__det)/self.__dim

    def map_to_elt(self, xi):
        """ maps ref coord xi to global coord x """
        return np.dot(self.__a, xi) + self.__b

    def map_to_ref(self, x):
        """ maps global coord x to local coord xi of reference element """
        return np.linalg.solve(self.__a, x - self.__b)

    def integrate(self, f, gauss=4):
        """ integrate (global) function f over element
        Input:
            f - function (callable) or constant (not callable) to be integrated
            gauss - degree of Gaussian quadrature
        Output:
            integral
        """

        # get quadrature points and weights
        xw = np.array(gaussPts(gauss, self.__dim))
        nq = xw.shape[0]             # number of quad points
        qweights = xw[:,2]           # quad weights
        qpoints = xw[:,:2]           # quad points

        # calculate and return integrals
        temp = self.measure()*(self.__dim/2.)
        if callable(f):
            return sum([ f( self.map_to_elt(qpoints[n]) )*qweights[n] for n in xrange(nq) ])*temp
        else:
            return sum([ f*qweights[n] for n in xrange(nq) ])*temp

    def jacobi(self):
        """ return Jacobi matrix of transformation x -> xi, i.e. dxi/dx """
        if self.__dim == 2:
            return self.__a_inv
        else:
            return 2./self.measure()

    def eval(self, n, x, m=0):
        """ evaluate shape function at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
            m - m'th method """
        method = self.method(m); xi = self.map_to_ref(x)
        if method == 'lagrange':
            return self.__sfns[m].eval(n, xi, derivative=False)
        elif method == 'rt':
            return np.dot(self.__a, self.__sfns[m].eval(n, xi, derivative=False))/self.__det

    def deval(self, n, x, m=0):
        """ evaluate shape function derivative at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
            m - m'th method """
        method = self.method(m); xi = self.map_to_ref(x)
        if method == 'lagrange':
            return np.dot(self.__sfns[m].eval(n, xi, derivative=True),self.jacobi())
        elif method == 'rt':
            return self.__sfns[m].eval(n, xi, derivative=True)/self.__det

    def evaluate(self, n, x, derivative=False, m=0):
        """ evaluate shape functions """
        if derivative:
            return self.deval(n, x, m)
        else:
            return self.eval(n, x, m)

    def assemble(self, c=None, gauss=4, derivative=[False,False], n=0, m=0):
        """
        integrate shape function products over this element
        Input:
            c - coefficient (variable or constant) can be tensor valued if 2D
            gauss - degree of Gaussian quadrature
            derivative - True if derivative of shape function evaluated
            n, m - methods
        Output:
            A - matrix of integrals
        """
        A = np.zeros((self.n_dofs(n), self.n_dofs(m)))
        for i in xrange(self.n_dofs(n)):
            for j in xrange(self.n_dofs(m)):
                if c is None:
                    A[i,j] = self.integrate(lambda x: np.dot(self.evaluate(i,x,derivative[0],n),\
                                                             self.evaluate(j,x,derivative[1],m)), gauss)
                elif not callable(c):
                    A[i,j] = self.integrate(lambda x: np.dot(np.dot(c, self.evaluate(i,x,derivative[0],n)),\
                                                                       self.evaluate(j,x,derivative[1],m)), gauss)
                else:
                    A[i,j] = self.integrate(lambda x: np.dot(np.dot(c(x), self.evaluate(i,x,derivative[0],n)),\
                                                                          self.evaluate(j,x,derivative[1],m)), gauss)
        return A

    def stiffness(self, c=None, gauss=4, m=0):
        """ assemble element stiffness matrix """
        return self.assemble(c, gauss, [True,True], m, m)

    def mass(self, c=None, gauss=4, m=0):
        """ assemble element mass matrix """
        return self.assemble(c, gauss, [False, False], m, m)

    def rhs(self, f, gauss=4, m=0):
        """ assemble element right hand side """
        if callable(f):
            return np.array([self.integrate(lambda x: f(x)*self.eval(n,x,m), gauss) for n in xrange(self.n_dofs(m))])
        else:
            return np.array([self.integrate(lambda x: self.eval(n,x,m), gauss)*f for n in xrange(self.n_dofs(m))])

    def __mul__(self, other):
        """ multiplication of two elements. They should be defined on same mesh """
        method = self.method() + other.method(); deg = self.deg() + other.deg()
        return FiniteElement(method, deg)

#--------------------------------------------------------------------------------------#

class RTElt:
    """ Raviart-Thomas element """
    def __init__(self, mesh):
        nodes, cells, boundary = mesh.get_data(1)
        n_nodes = nodes.shape[0]
        n_cells = mesh.n_cells()

        """ nodes_to_element[k,l] = j, where
        k,l - node numbers of edge
        j - cell no of cell with this edge (orientation respected -> counter clockwise) """
        nodes_to_element = np.zeros((n_nodes, n_nodes), dtype=int)
        temp = np.zeros((n_nodes, n_nodes), dtype=int)
        for n in xrange(n_cells):
            nodes_to_element[cells[n,:].reshape((3,1)), cells[n,[1,2,0]]] += n*np.identity(3)
            temp[cells[n,:].reshape((3,1)), cells[n,[1,2,0]]] += (n+1)*np.identity(3) # <-- because in matlab idex start from 0

        self.__nodes_to_element = nodes_to_element

        """ nodes_to_edge[k,l] = j, where
        k,l - node numbers of edge
        j - edge no of edge with nodes k,l
        """
        B = temp + temp.T
        # I,J = np.nonzero(np.triu(B))
        J,I = np.nonzero(np.triu(B).T) # <------ this is as the original
        nodes_to_edge = np.zeros((n_nodes, n_nodes), dtype=int)
        n_edges = len(I)
        for k in xrange(n_edges):
            nodes_to_edge[I[k], J[k]] = k
        nodes_to_edge = nodes_to_edge + nodes_to_edge.T

        self.__n_edges = n_edges
        self.__nodes_to_edge = nodes_to_edge

        """ edge_to_element[j,[2,3]] = [m,n], where
        if common edge j belongs to elts m and n
        """
        edge_to_element = -np.ones((n_edges,4),dtype=int)
        for m in xrange(n_cells):
            for k in xrange(1,4):
                i = cells[m, k - 1]; j = cells[m, k % 3]
                p = nodes_to_edge[i, j]

                if edge_to_element[p,0] == -1:
                    edge_to_element[p,:] = [i, j, temp[i,j]-1, temp[j,i]-1]

        self.__edge_to_element = edge_to_element

        # To produce the interior and exterior edges
        interior_edges = edge_to_element[np.where(edge_to_element[:,3] >= 0)[0],:]

        temp = np.array([np.where(edge_to_element[:,3] == -1)[0]]).T
        exterior_edges = edge_to_element[temp, [0,1,2]]

        self.__interior_edges = interior_edges
        self.__exterior_edges = exterior_edges

    def n_edges(self):
        return self.__n_edges

    def nodes_to_element(self):
        return self.__nodes_to_element

    def nodes_to_edge(self):
        return self.__nodes_to_edge

    def edge_to_element(self):
        return self.__edge_to_element

    def interior_edges(self):
        return self.__interior_edges

    def exterior_edges(self):
        return self.__exterior_edges


if __name__ == '__main__':
    from meshing import UnitSquareMesh, UnitIntMesh
    from shapefuncs import ShapeFunctions

    def affine_mapping_test():
        """ affine mapping test """

        """ ---------------- 2D test ---------------- """
        mesh = UnitSquareMesh(8,8)
        vs = mesh.elt_to_vcoords()
        fe = FiniteElement()
        fe.initialize(mesh.dim())

        # 2D map to elt test
        for v in vs:
            fe.set_vertices(v)
            v1 = fe.map_to_elt([0,0]); v2 = fe.map_to_elt([1,0]); v3 = fe.map_to_elt([0,1])
            assert np.allclose(v - np.array([v1, v2, v3]),0), '2D map to elt failed.'

        # 2D map to ref test
        for v in vs:
            fe.set_vertices(v)
            v1 = fe.map_to_ref(v[0]); v2 = fe.map_to_ref(v[1]); v3 = fe.map_to_ref(v[2])
            assert np.allclose(np.array([v1, v2, v3]) - np.array([[0,0], [1,0], [0,1]]),0), '2D map to ref failed.'

        """ ---------------- 2D edge ---------------- """
        nodes_to_coord, elts_to_nodes = mesh.get_data()
        boundary = mesh.get_bdata()
        bv = mesh.bvertices()
        fe.initialize(1)

        # 2D map to edge test
        for key in bv.keys():
            vs = bv[key]
            for v in vs:
                fe.set_vertices(v)
                assert np.allclose(v - [fe.map_to_elt([-1,0]), fe.map_to_elt([1,0])],0), '2D map to edge ({}) failed.'.format(key)

        # 2D edge to ref test
        for key in bv.keys():
            vs = bv[key]
            for v in vs:
                fe.set_vertices(v)
                assert np.allclose(np.array([[-1,0],[1,0]]) - [fe.map_to_ref(v[0]), fe.map_to_ref(v[1])],0),\
                 '2D edge to ref ({}) failed.'.format(key)

        """ ---------------- 1D test ---------------- """
        mesh = UnitIntMesh(8)
        vs = mesh.elt_to_vcoords()
        fe = FiniteElement()
        fe.initialize(mesh.dim())

        # 1D map to elt test
        for v in vs:
            fe.set_vertices(v)
            assert np.allclose(v - [fe.map_to_elt([-1,0]), fe.map_to_elt([1,0])],0), '1D map to elt failed.'

        # 1D map to ref test
        for v in vs:
            fe.set_vertices(v)
            assert np.allclose(np.array([[-1,0],[1,0]]) - [fe.map_to_ref(v[0]), fe.map_to_ref(v[1])],0), '1D map to ref failed.'

        print "Affine mapping test OK."

    #affine_mapping_test()

    def measure_test():
        """ finite element measure test """
        n = 3; tri_measure = pow(1./n,2)/2; int_measure = 1./n

        """ ---------------- 2D test ---------------- """
        mesh = UnitSquareMesh(n,n)
        fe = FiniteElement()
        fe.initialize(mesh.dim())
        vs = mesh.elt_to_vcoords()

        # 2D measure test
        for v in vs:
            fe.set_vertices(v)
            assert np.isclose(fe.measure() - tri_measure,0), '2D measure test failed.'

        """ ---------------- 2D edge ---------------- """
        nodes_to_coord, elts_to_nodes = mesh.get_data()
        boundary = mesh.get_bdata()
        bv = mesh.bvertices()
        fe.initialize(1)

        # 2D edge measure test
        for key in bv.keys():
            vs = bv[key]
            for v in vs:
                fe.set_vertices(v)
                assert np.isclose(fe.measure() - int_measure,0), '2D edge measure test ({}) failed.'.format(key)
                # det = abs(np.linalg.det(fe._FiniteElement__a))
                # assert np.isclose(det - int_measure,0), '2D edge determinant test ({}) failed.'.format(key)

        """ ---------------- 1D test ---------------- """
        mesh = UnitIntMesh(n)
        fe = FiniteElement()
        fe.initialize(mesh.dim())
        vs = mesh.elt_to_vcoords()

        # 1D measure test
        for v in vs:
            fe.set_vertices(v)
            assert np.isclose(fe.measure() - int_measure,0), '1D measure test failed.'
            # det = abs(np.linalg.det(fe._FiniteElement__a))
            # assert np.isclose(det - int_measure,0), '1D determinant test failed.'

        print "Measure test OK."

    #measure_test()

    def boundary_map_test():
        """ boundary mapping test """
        mesh = UnitSquareMesh(4,4)
        fe = FiniteElement()
        fe.initialize(1)
        bvertices = mesh.bvertices()

        for name in bvertices.keys():
            vertices = np.array(bvertices[name])

            # map to ref
            for i in xrange(len(vertices)):
                vertex = vertices[i]
                fe.set_vertices(vertex)
                left = fe.map_to_ref(vertex[0])
                right = fe.map_to_ref(vertex[1])
                mid = fe.map_to_ref((vertex[0] + vertex[1])/2)

                assert np.allclose(np.array([[-1,0], [1,0]]) - [left, right],0) and np.allclose(np.array([0,0]) - mid,0)
                #print "interval: [{}, {}].  mid: {}".format(left[0], right[0], mid[0])

            # map to elt
            for i in xrange(len(vertices)):
                vertex = vertices[i]
                fe.set_vertices(vertex)
                left = fe.map_to_elt([-1,0])
                mid = fe.map_to_elt([0,0])
                right = fe.map_to_elt([1,0])

                m = sum(vertex)/2
                assert np.allclose(vertex - [left, right],0) and np.allclose(m - mid,0)

        print "Boundary mapping test OK."

    #boundary_map_test()

    def boundary_integral_test():
        """ boundary integral test """
        import math
        mesh = UnitSquareMesh(1,1)
        boundary = mesh.get_bdata()
        bvertices = mesh.bvertices()

        # bottom
        name = 'bottom'
        vertices = bvertices[name]
        fe = FiniteElement(); fe.initialize(1)
        fe.set_vertices(vertices[0])

        f = lambda x: x[0]**2; exact = [1./12, 1./4]
        #f = lambda x: math.sin(x[0]); exact = [1-math.sin(1), math.sin(1) - math.cos(1)]
        rhs = fe.rhs(f)
        assert np.allclose(rhs - exact,0)

        # right
        name = 'right'
        vertices = bvertices[name]
        fe.set_vertices(vertices[0])
        #f = lambda x: x[1]**2; exact = [1./12, 1./4]
        f = lambda x: math.sin(x[1]); exact = [1-math.sin(1), math.sin(1) - math.cos(1)]
        rhs = fe.rhs(f)
        assert np.allclose(rhs - exact,0)

        # top
        name = 'top'
        vertices = bvertices[name]
        fe.set_vertices(vertices[0])
        #f = lambda x: x[0]**2; exact = [1./4, 1./12]
        f = lambda x: math.sin(x[0]); exact = [math.sin(1) - math.cos(1), 1-math.sin(1)]
        rhs = fe.rhs(f)
        assert np.allclose(rhs - exact,0)

        # left
        name = 'left'
        vertices = bvertices[name]
        fe.set_vertices(vertices[0])
        #f = lambda x: x[1]**2; exact = [1./4, 1./12]
        f = lambda x: math.sin(x[1]); exact = [math.sin(1) - math.cos(1), 1-math.sin(1)]
        rhs = fe.rhs(f)
        assert np.allclose(rhs - exact,0)

        print "Boundary integral test OK."

    #boundary_integral_test()

    def rt_test():
        mesh = UnitSquareMesh(2,2)
        #mesh.plot()
        rt = RTElt(mesh)
        n2e = rt.nodes_to_element()

        assert rt.n_edges() == 16, 'n_edges failed.'

        ext_edges = [[0,1], [1,2], [2,5], [5,8], [8,7], [7,6], [6,3], [3,0]]
        n2e_ee_fasit = [0, 2, 2, 6, 7, 5, 5, 1]

        int_edges = [[0,4], [1,4], [1,5], [3,4], [4,5], [3,7], [4,7], [4,8]]
        n2e_ie_fasit = [1, 0, 3, 4, 6, 5, 4, 7]

        i = 0
        for ee in ext_edges:
            assert n2e[ee[0], ee[1]] == n2e_ee_fasit[i], "nodes to elt ext failed."
            i += 1

        i = 0
        for ie in int_edges:
            assert n2e[ie[0], ie[1]] == n2e_ie_fasit[i], "nodes to elt int failed"
            i += 1

        """ ------------- nodes to element seems to be OK ------------- """

        n2e = rt.nodes_to_edge()

        n2e_ee_fasit = [0, 1, 7, 14, 15, 12, 9, 2]
        n2e_ie_fasit = [3, 4, 6, 5, 8, 10, 11, 13]

        i = 0
        for ee in ext_edges:
            assert n2e[ee[0], ee[1]] == n2e_ee_fasit[i], "nodes to edge ext failed."
            i += 1

        i = 0
        for ie in int_edges:
            assert n2e[ie[0], ie[1]] == n2e_ie_fasit[i], "nodes to edge int failed."
            i += 1

        """ ------------- nodes to edge seems to be OK ------------- """

        e2e = rt.edge_to_element(); n_edges = rt.n_edges()

        edges = [[0,1], [1,2], [0,3], [0,4], [1,4], [3,4], [1,5],\
                 [2,5], [4,5], [3,6], [3,7], [4,7], [6,7], [4,8], [5,8], [7,8]]

        e2e_fasit = [[0,-1], [2,-1], [1,-1], [0,1], [0,3], [1,4], [2,3], [2,-1],\
                     [3,6], [5,-1], [4,5], [4,7], [5,-1], [6,7], [6,-1], [7,-1]]

        for j in xrange(n_edges):
            assert all(e2e[j,[2,3]] == e2e_fasit[j]), 'edge to element failed.'

        """ ------------- edge to element seems to be OK ------------- """

        #print e2e
        #print rt.interior_edges()
        #print rt.exterior_edges()

    #rt_test()

#############################################################################
    def random_edge_test():
        cell = [0]; vertex = np.array([[1,1],[2,2]],dtype=float)
        sfns = Lagrange(0,1)
        fe = LagrangeElt(cell, vertex, sfns)
        print "map v to ref: ", fe.map_to_ref(vertex[0]), fe.map_to_ref(vertex[1])
        print "map c to ref: ", fe.map_to_ref(np.array([1.5,1.5]))
        print "map to ref: ", [-1.] - fe.map_to_ref(vertex[0]), [1.] - fe.map_to_ref(vertex[1])
        print "map to elt: ", vertex[0] - fe.map_to_elt([-1.]), vertex[1] - fe.map_to_elt([1.])

    #random_edge_test()

    def edge_mapping_test():
        mesh = UnitSquareMesh()
        mesh.set_degree(0)

        sfns = Lagrange(0,1)

        names = mesh.boundary_names(True)
        for name in names:
            print "\n" + name.upper() + ":\n"
            cells = mesh.boundary(name); vertices = mesh.edge_vertices(name)
            for n in range(len(cells)):
                cell = cells[n]; vertex = vertices[n]
                print "cell {}: {}".format(n, cell)
                print "vertices:\n", vertex
                fe = LagrangeElt(cell, vertex, sfns)
                print "map to ref: ", [-1.] - fe.map_to_ref(vertex[0]), [1.] - fe.map_to_ref(vertex[1])
                print "map to elt: ", vertex[0] - fe.map_to_elt([-1.]), vertex[1] - fe.map_to_elt([1.])
                print "measure: ", fe.measure()
                print "assemble: ", fe.assemble(gauss=4)
                print "rhs: ", fe.rhs(1,gauss=4)
                print "integrate: ", fe.integrate(1,gauss=4)
                print "-------------------------------------------------------------------------"

    #edge_mapping_test()

    def mapping_test():
        mesh = UnitIntMesh(20); mesh.set_degree(0)
        sfns = Lagrange(0,1)

        for n in range(mesh.n_cells()):
            cell = mesh.cells(n); vertices = mesh.vertices(n)
            print "cell {}: {}".format(n, cell)
            print "vertices:\n", vertices
            fe = LagrangeElt(cell, vertices, sfns)
            print "map to ref: ", [-1.] - fe.map_to_ref(vertices[0]), [1.] - fe.map_to_ref(vertices[1])
            print "map to elt: ", vertices[0] - fe.map_to_elt([-1.]), vertices[1] - fe.map_to_elt([1.])
            print "measure: ", fe.measure()
            print "assemble: ", fe.assemble(gauss=4)
            print "rhs: ", fe.rhs(1,gauss=4)
            print "integrate: ", fe.integrate(1,gauss=4)
            print "-------------------------------------------------------------------------"

    #mapping_test()


    def edge_intergral_test2():
        import math; pi = math.pi

        square = UnitSquareMesh(16,16)
        bottom = square.bottom(); bottomv = square.edge_vertices('bottom')
        right = square.right(); rightv = square.edge_vertices('right')
        top = square.top(); topv = square.edge_vertices('top')
        left = square.bottom(); leftv = square.edge_vertices('left')

        intv = UnitIntMesh(16)
        sfns = Lagrange(1,1)

        gauss = 8

        # bottom test
        ints1 = []; ints2 = []
        f = lambda x: math.sin(pi*x[0]); exact = 2/pi
        for n in range(intv.n_cells()):
            fe1 = LagrangeElt(intv.cells(n), intv.vertices(n), sfns)
            ints1.append(fe1.integrate(f,gauss))

            fe2 = LagrangeElt(bottom[n], bottomv[n], sfns)
            ints2.append(fe2.integrate(f,gauss))

        ints1 = np.array(ints1)
        ints2 = np.array(ints2)

        print "bottom:\n", ints1 - ints2
        print "check: ", exact-sum(ints1), exact-sum(ints2)

        # right test
        ints2 = []
        f = lambda x: math.sin(pi*x[1])
        for n in range(intv.n_cells()):
            fe2 = LagrangeElt(right[n], rightv[n], sfns)
            ints2.append(fe2.integrate(f,gauss))

        print "right:\n", ints1 - ints2
        print "check: ", exact-sum(ints1), exact-sum(ints2)

        # top test
        ints2 = []
        f = lambda x: math.sin(pi*x[0])
        for n in range(intv.n_cells()):
            fe2 = LagrangeElt(top[n], topv[n], sfns)
            ints2.append(fe2.integrate(f,gauss))
        ints2.reverse()

        print "top:\n", ints1 - ints2
        print "check: ", exact-sum(ints1), exact-sum(ints2)

        # left test
        ints2 = []
        f = lambda x: math.sin(pi*x[1])
        for n in range(intv.n_cells()):
            fe2 = LagrangeElt(left[n], leftv[n], sfns)
            ints2.append(fe2.integrate(f,gauss))
        ints2.reverse()

        print "left:\n", ints1 - ints2
        print "sums: ", exact-sum(ints1), exact-sum(ints2)

    #edge_intergral_test2()


    def integral_test():
        mesh = UnitSquareMesh(16,16)
        #mesh.set_degree(0)
        sfns = Lagrange(1,2)

        meas = []
        for n in range(mesh.n_cells()):
            cell = mesh.cells(n); vertices = mesh.vertices(n)
            fe = LagrangeElt(cell, vertices, sfns)
            a = fe.measure()
            meas.append(a)
            ass1 = fe.assemble(derivative=True)
            E = vertices[[1,2,0],:] - vertices[[2,0,1],:]
            ass2 = np.dot(E,E.T)/(4*a)
            print "\n-----------------------------------------"
            print "-----------------------------------------"
            temp = ass1 - ass2; temp2 = sum(sum(temp))
            print temp
            print "\n", temp2
            print np.isclose(temp2, 0)

            print "\n-----------------------------------------"
            #f = lambda x: x[0] + x[1]
            f = lambda x: 1.
            c = [sum(vertices[:,0]), sum(vertices[:,1])]
            rhs = fe.rhs(f); rhs2 = f(c)*a/3
            print rhs
            print rhs2
            print np.isclose(sum(rhs-rhs2),0)

        print "measure of domain: ", sum(meas)


    #integral_test()
