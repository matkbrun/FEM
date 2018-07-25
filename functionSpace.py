import numpy as np
from numpy.linalg import cond
from scipy.sparse import csc_matrix, dia_matrix
from finiteElement import FiniteElement
from shapefuncs import ShapeFunctions

class FunctionSpace:
    """ function space class """
    def __init__(self, mesh, fe, gauss=4):
        """
        Input:
            mesh - mesh it is built on
            fe - Finite Element (mixed or not)
            gauss - degree of Gaussian quadrature
        """

        # get data from finite element
        deg = fe.deg()
        method = fe.method()
        mixed = fe.mixed()
        n_FE = len(method)

        # get data from mesh
        self.__gauss = gauss                                # degree of Gaussian quadrature
        self.__dim = mesh.dim()                             # dimension (same as mesh)
        self.__n_elts = mesh.n_elts()                       # number of elements
        self.__elt_to_vcoords = mesh.elt_to_vcoords()       # element number to vertex coords matrix

        # get data from mesh corresponding to FEM methods given
        node_to_coords = []; elt_to_nodes = []; n_nodes = []
        for n in xrange(n_FE):
            if method[n] == 'lagrange':
                ntc, etn = mesh.get_data(deg[n])
            elif method[n] == 'rt':
                ntc, etn = mesh.get_data(1)
            n_nodes.append(ntc.shape[0])
            node_to_coords.append(ntc); elt_to_nodes.append(etn)

        # set data
        self.__deg = deg                            # degree of shape functions
        self.__method = method                      # name of FEM methods
        self.__mixed = mixed                        # True if mixed FE
        self.__node_to_coords = node_to_coords      # node number to coord matrix (one for each FE)
        self.__elt_to_nodes = elt_to_nodes          # element number to node numbers matrix (one for each FE)
        self.__n_nodes = n_nodes                    # number of nodes (one for each FE)
        self.__fe = fe                              # finite element
        self.__n_FE = n_FE                          # number of FE (1 if not mixed)
        self.mesh = mesh                            # mesh

    def mixed(self):
        """ return True if mixed function space """
        return self.__mixed

    def n_FE(self):
        """ return number of FE in this space """
        return self.__n_FE

    def dim(self):
        """ return spatial dimension """
        return self.__dim

    def node_to_coords(self, m=0):
        """ return node number to coordinate matrix for FE number m """
        return self.__node_to_coords[m]

    def n_nodes(self, m=None):
        """ return total number of nodes, or number of nodes for FE number m """
        if m is None:
            return sum(self.__n_nodes)
        else:
            return self.__n_nodes[m]

    def elt_to_nodes(self, m=0):
        """ return element number to node number matrix (of FE number m, in case of mixed space) """
        return self.__elt_to_nodes[m]

    def n_elts(self):
        """ return number of elements """
        return self.__n_elts

    def assemble(self, c=None, derivative=[False,False], n=0, m=0):
        """ assemble linear system
        Input:
            c - coefficient (variable or constant), can be tensor valued if appropriate
            derivative - True if derivative of shape function
            n,m - FE numbers n and m (in case of mixed space)
        Output:
            A - (n,m) matrix (dense if mixed, sparse if not)
        """

        # initialize finite element
        self.__fe.initialize(self.dim())

        # assemble
        print "Assembling linear system..."
        A = np.zeros((self.n_nodes(n), self.n_nodes(m)))            # global assembly
        for j in xrange(self.__n_elts):
            # assemble element j
            self.__fe.set_vertices(self.__elt_to_vcoords[j])
            elt_assemble = self.__fe.assemble(c, self.__gauss, derivative, n, m)

            local_nodes_n = np.array([self.elt_to_nodes(n)[j]])     # row indices
            local_nodes_m = self.elt_to_nodes(m)[j]                 # column indices
            A[local_nodes_n.T, local_nodes_m] += elt_assemble       # add to global assembly

        #print "Condition number: ", cond(A)
        if self.__mixed:
            return A
        else:
            return csc_matrix(A)
            #return csr_matrix(A)
            #return dia_matrix(A)

    def stiffness(self, c=None, m=0):
        """ assemble stiffness matrix (of FE number m,in case of mixed space) """
        return self.assemble(c, [True,True], m, m)

    def mass(self, c=None, m=0):
        """ assemble mass matrix (of FE number m, in case of mixed space) """
        return self.assemble(c, [False,False], m, m)

    def rhs(self, f, m=0):
        """ assemble right hand side vector (of FE number m, in case of mixed space)
        Input:
            f - function (or constant)
            m - if mixed space, gives number of FE
        """
        rhs = np.zeros(self.n_nodes(m))
        if f == 0:
            return rhs
        else:
            self.__fe.initialize(self.dim())
            elt_to_nodes = self.elt_to_nodes(m)
            for j in xrange(self.n_elts()):
                self.__fe.set_vertices(self.__elt_to_vcoords[j])
                rhs[elt_to_nodes[j]] += self.__fe.rhs(f, self.__gauss, m)
            return rhs

    def belt_to_nodes(self, *args):
        """ get node number of boundary elements
        Input:
            args: (optional)
                m (int) - if mixed space, gives number of FE (default: 0)
                names (strings) - names for boundary dictionary (default: entire boundary)
        Output:
            boundary element number to node numbers matrix
        """
        # check args input
        if len(args) == 0:
            m = 0; names = []
        elif isinstance(args[0], int):
            m = args[0]; names = [args[k] for k in range(1,len(args))]
        else:
            m = 0; names = args

        # get boundary data
        if self.__method[m] == 'lagrange':
            boundary = self.mesh.get_bdata(self.__deg[m])
        elif self.__method[m] == 'rt':
            boundary = self.mesh.get_bdata(1)

        # return boundary element number to node numbers matrix
        if len(names) == 0:
            return np.array([b for k in boundary.keys() for b in boundary[k]], dtype=int)
        else:
            return np.array([b for k in names for b in boundary[k]], dtype=int)

    def edge_to_vcoords(self, *names):
        """ get vertex coords of boundary edges (only for 2D space)
        Input:
            names - names for boundary dictionary
        Output:
            edge number to vertex coords matrix
        """
        assert self.__dim == 2, 'Edge vertices only available for 2D function space.'

        # get bondary edge dictionary
        bvertices = self.mesh.bvertices()

        # return edge number to vertex coords matrix
        if len(names) == 0:
            return np.array([v for k in bvertices.keys() for v in bvertices[k]])
        else:
            return np.array([v for k in names for v in bvertices[k]])

    def assemble_boundary(self, g, *args):
        """ assemble over boundary in case of Neuman boundary conditions (only for 2D space)
        Input:
            g - function (or constant) to be integrated with shape functions over boundary
            args: (optional)
                m (int) - if mixed space, gives number of FE (default: 0)
                names (strings) - names for boundary dictionary (default: entire boundary)
        Output:
            array that can be added to RHS vector to incorporate Neuman boundary conditions
        """
        assert self.__dim == 2, 'Edge integration only for 2D function space.'

        # check args input
        if len(args) == 0:
            m = 0; names = []
        elif isinstance(args[0], int):
            m = args[0]; names = [args[k] for k in range(1,len(args))]
        else:
            m = 0; names = args

        # get boundary data
        edge_to_nodes = self.belt_to_nodes(m, *names)
        edge_to_vcoords = self.edge_to_vcoords(*names)

        # assemble boundary
        fe = FiniteElement(self.__method[m], self.__deg[m])
        fe.initialize(1)
        b = np.zeros(self.n_nodes(m))
        for n in xrange(len(edge_to_nodes)):
            fe.set_vertices(edge_to_vcoords[n])
            b[edge_to_nodes[n]] += fe.rhs(g, self.__gauss, m)
        return b

if __name__ == '__main__':
    from scipy.sparse.linalg import spsolve, cg
    import math
    from meshing import UnitSquareMesh, UnitIntMesh, Gmesh
    from boundaryCond import Dirichlet
    from aux import fem_solver, plot_sol

    def p0_test():
        """ test with P0 element """
        u_ex = lambda x: 16.*x[0]*(1.-x[0])*x[1]*(1.-x[1])      # exact solution

        mesh = UnitSquareMesh(16,16)
        fs = FunctionSpace(mesh, FiniteElement('lagrange', 0), gauss=4)

        A = fs.mass()
        rhs = fs.rhs(u_ex)

        #u = spsolve(A,rhs)
        #bc = Dirichlet(fs)
        u = fem_solver(fs, A, rhs, bcs=bc)
        plot_sol(u,fs.node_to_coords(),u_ex)

    #p0_test()

    def mixed_test():
        """ Poisson problem 2D test with Neuman bc enforced by Lagrange multiplier,
        using mixed P1-P0 elements:
        ------------------------------------------------------------------
            find (u,c) such that
            (grad u, grad v) + (c,v) = (f,v) + (g,v)_gamma, forall v (P1)
                               (u,d) = 0,                   forall d (P0)
        ------------------------------------------------------------------
         """
        g = lambda x: - math.sin(math.pi*x[0])                                          # Neuman bc
        f = lambda x: 10*math.exp(-(pow(x[0] - .5, 2) + pow(x[1] - .5, 2)) / .02 )      # right hand side

        mesh = UnitSquareMesh(32,32)                                        # mesh
        me = FiniteElement('lagrange', 1)*FiniteElement('lagrange', 0)      # mixed element
        fs = FunctionSpace(mesh, me, gauss=4)                               # mixed space

        Auu = fs.stiffness()                                    # u stiffness matrix
        Auc = fs.assemble(derivative=[False,False],n=0,m=1)     # mixed mass matrix
        rhs_u = fs.rhs(f) + fs.assemble_boundary(g)             # right hand side for u

        n = fs.n_nodes(0); m = fs.n_nodes(1); n_nodes = n+m
        print "n = {} (Solution), m = {} (Lagrange)".format(n,m)
        print "Auu: ", Auu.shape
        print "Auc: ", Auc.shape

        # assemble global linear system
        A = np.zeros((n_nodes, n_nodes))
        A[:n,:n] = Auu
        A[:n,n:] = Auc
        A[n:,:n] = Auc.T

        rhs = np.zeros(n_nodes)
        rhs[:n] = rhs_u

        u,c = fem_solver(fs, A, rhs, CG=True)
        print "u: ", u.shape
        print "c: ", c.shape

        plot_sol(u, fs.node_to_coords(0), contour=True, name='Solution')
        plot_sol(c, fs.node_to_coords(1), contour=True, name='Lagrange')

    #mixed_test()

    def poisson_test_2d():
        """ Poisson problem 2D test with homogenous Dirichlet bc """
        u_ex = lambda x: 16.*x[0]*(1.-x[0])*x[1]*(1.-x[1])      # exact solution
        f = lambda x: 32.*(x[1]*(1.-x[1]) + x[0]*(1.-x[0]))     # right hand side

        mesh = UnitSquareMesh(16,16,diag='right')
        #mesh = Gmesh('Meshes/square_P2.msh')
        fs = FunctionSpace(mesh, FiniteElement('lagrange', deg=1), gauss=4)

        A = fs.stiffness()
        rhs = fs.rhs(f)

        bc = Dirichlet(fs)
        u = fem_solver(fs, A, rhs, bc)
        plot_sol(u,fs.node_to_coords(),u_ex)

        # --- manual incorporation of Dirichlet bc ---
        # u = np.zeros(fs.n_nodes())
        # belt_to_nodes = fs.belt_to_nodes()
        # free_nodes = np.setdiff1d(range(fs.n_nodes()), np.unique(belt_to_nodes))
        # n_free_nodes = len(free_nodes)

        # u[free_nodes] = spsolve(A[free_nodes.reshape(n_free_nodes, 1), free_nodes], rhs[free_nodes])
        # plot_sol(u,fs.node_to_coords(),u_ex)

    #poisson_test_2d()

    def poisson_test_1d():
        """ Poisson problem 1D test with homogenous Dirichlet bc """
        u_ex = lambda x: 4.*x[0]*(1.-x[0])      # exact solution
        f = 8.                                  # right hand side

        mesh = UnitIntMesh(32)
        fs = FunctionSpace(mesh, FiniteElement('lagrange', 1), gauss=4)
        A = fs.stiffness()
        rhs = fs.rhs(f)

        bc = Dirichlet(fs)
        u = fem_solver(fs, A, rhs, bc)
        plot_sol(u,fs.node_to_coords(),u_ex)

        # --- manual incorporation of Dirichlet bc ---
        # boundary = fs.belt_to_nodes()
        # u = np.zeros(fs.n_nodes())
        # free_nodes = np.setdiff1d(range(fs.n_nodes()), np.unique(boundary))
        # n_free_nodes = len(free_nodes)

        # u[free_nodes] = spsolve(A[free_nodes.reshape(n_free_nodes, 1), free_nodes], rhs[free_nodes])
        # plot_sol(u,fs.node_to_coords(),u_ex)

    #poisson_test_1d()
