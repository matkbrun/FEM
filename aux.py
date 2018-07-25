import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix, issparse
from scipy.sparse.linalg import spsolve, cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from boundaryCond import Dirichlet
from functionSpace import FunctionSpace

"""
------------------------------------------------------------------------------------
Some auxiliary useful functions
    fem_solver - solves linear system with Dirichlet boundary conditions
    plot_piecewise - makes piecewise plot of 1d solution (for P0 elts)
    plot_sol - plots solution, 1d or 2d
------------------------------------------------------------------------------------
"""
def inc_dirichlet(fs, bcs):
    """ help incorporate Dirichlet bc
    Input:
        fs - function space
        bcs - Dirichlet boundary conditions
    Output:
        u - updated solution vector
        dirichlet_nodes - unique node numbers of node involved in bcs
    """
    # initialize solution vector
    u = np.zeros(fs.n_nodes())

    # put unique Dirichlet nodes here
    dirichlet_nodes = np.array([], dtype=int)

    # inc bc in solution vector and save nodes
    for bc in bcs:
        unique_nodes = bc.unique_nodes()                            # unique node numbers in this bc
        new_nodes = np.setdiff1d(unique_nodes, dirichlet_nodes)     # new nodes not in previous bc
        u[new_nodes] += bc.assemble()[new_nodes]                    # update solution vector
        dirichlet_nodes = np.concatenate((dirichlet_nodes, new_nodes))

    return u, dirichlet_nodes

def inc_mixed_dirichelt(fs, bcs):
    """ help incorporate Dirichlet bc for mixed space
    Input:
        fs - mixed function space
        bcs - Dirichlet boundary conditions
    Output:
        u - updated solution vector
        dirichlet_nodes - unique node numbers of node involved in bcs
    """
    # initialize solution vector
    u = np.zeros(fs.n_nodes())

    # put unique Dirichlet nodes here
    dirichlet_nodes = np.array([], dtype=int)

    # get indexes of FE for each bc
    ms = np.unique([bc.m() for bc in bcs])

    # inc bc in solution vector and save nodes
    for m in ms:
        u_m = np.zeros(fs.n_nodes(m))
        dnodes_m = np.array([], dtype=int)
        for bc in bcs:
            if bc.m() == m:
                unique_nodes = bc.unique_nodes()
                new_nodes = np.setdiff1d(unique_nodes, dnodes_m)
                u_m[new_nodes] = bc.assemble()[new_nodes]
                dnodes_m = np.concatenate((dnodes_m, new_nodes))
        temp = sum([fs.n_nodes(j) for j in xrange(m)])
        dnodes_m += temp
        u[dnodes_m] = u_m
        dirichlet_nodes = np.concatenate((dirichlet_nodes, dnodes_m))

    return u, dirichlet_nodes

def fem_solver(fs, A, rhs, bcs=None, CG=False, tol=1e-05):
    """
    Solve linear system Au = rhs with specified Dirichlet boundary conditions (optional)
    Input:
        fs - function space
        A - linear system (sparse if not mixed, dense if mixed)
        rhs - right hand side vector
        bcs - Dirichlet boundary conditions (optional)
        cg - True if CG for solving linear system
        tol - tolerance of CG (optional)
    Output:
        u - solution vector, if not mixed
        u_1,...,u_n - if mixed
    """

    if bcs is not None: # <------------------------------- incorporate Dirichlet boundary conditions (if any)
        # put bcs in correct format
        if not isinstance(bcs, list):
            bcs = [bcs]

        # get updated solution vector and unique node numbers
        if fs.mixed():
            u, dirichlet_nodes = inc_mixed_dirichelt(fs, bcs)

        else:
            u, dirichlet_nodes = inc_dirichlet(fs, bcs)

        free_nodes = np.setdiff1d(range(fs.n_nodes()), dirichlet_nodes)     # nodes not used by any Dirichlet bc
        n_free_nodes = len(free_nodes)                                      # number of free nodes

        # convert linear system to dense (if needed)
        if issparse(A):
            A = A.todense()

        # modify linear system
        A_free = A[free_nodes.reshape(n_free_nodes, 1), free_nodes]             # part of A not part of Dirichlet bc
        A_dirichlet = A[free_nodes.reshape(n_free_nodes, 1), dirichlet_nodes]   # part of A part of Dirichlet bc

        rhs = (rhs[free_nodes] - A_dirichlet.dot(u[dirichlet_nodes])).reshape((n_free_nodes,1))

        # convert to sparse
        A_free = csc_matrix(A_free)
        #A_free = csr_matrix(A_free)
        #A_free = dia_matrix(A_free)

        # solve
        if CG:
            temp, info = cg(A_free, rhs, tol=tol)
            assert info == 0, 'CG method failed to converge.'
            u[free_nodes] = temp
        else:
            u[free_nodes] = spsolve(A_free, rhs)

    else: # <--------------------------------------------- no Dirichlet boundary conditions
        # convert linear system to sparse (if needed)
        if not issparse(A):
            A = csc_matrix(A)
            #A = csr_matrix(A)
            #A = dia_matrix(A)

        # solve
        if CG:
            u, info = cg(A, rhs, tol=tol)
            assert info == 0, 'CG method failed to converge.'
        else:
            u = spsolve(A, rhs)

    # return solution vector(s)
    if fs.mixed():
        us = []; temp = 0
        for j in xrange(fs.n_FE()):
            n = fs.n_nodes(j)
            us.append(u[temp:n+temp])
            temp = n
        return us
    else:
        return u

def plot_sol(u, nodes, u_ex=None, contour=False, name=''):
    """ makes a plot of u on mesh
    Input:
        u - solution vector
        mesh - 1d or 2d mesh
        u_ex - exact solution (optional)
        name - name of figure
        contour - True if contour plot, 3D if not """

    # set up figure
    if name == '':
        name = 'Plotting'
    if u_ex is None:
        fig = plt.figure(num=name)
    else:
        fig = plt.figure(figsize=(10,6), num=name)

    #dim = nodes.shape[1]
    if all(nodes[:,1] == 0):
        dim = 1
    else:
        dim = 2

    if dim == 2:
        X = nodes[:,0]; Y = nodes[:,1]
        #triangulation = tri.Triangulation(X, Y)

        if u_ex is None:
            if not contour: # 3D plot
                ax = fig.gca(projection='3d')
                ax.plot_trisurf(X, Y, u, linewidth=0.2, antialiased=True, cmap=plt.cm.Spectral)
            else: # contour plot
                plot = plt.tricontourf(X,Y,u,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')

            plt.title("computed solution")
            plt.xlabel('x'); plt.ylabel('y')
            plt.xticks([]); plt.yticks([])
        else:
            # evaluate exact solution at nodes
            U_ex = [u_ex(node) for node in nodes]

            if not contour: # 3D plot
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.plot_trisurf(X, Y, u, linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                plt.title("computed solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.plot_trisurf(X, Y, U_ex, linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                plt.title("exact solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')
            else: # contour plot
                plt.subplot(121)
                plot = plt.tricontourf(X,Y,u,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')
                plt.title("computed solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

                plt.subplot(122)
                plot = plt.tricontourf(X,Y,U_ex,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')
                plt.title("exact solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

            # errors
            print "error:"
            print "\t|u - u_ex| = ", norm(U_ex-u)

    else: # dim = 1
        if u_ex is None:
            plt.plot(nodes[:,0], u)
            plt.title("computed solution")
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            plt.subplot(121)
            plt.plot(nodes[:,0], u)
            plt.title("computed solution")
            plt.xlabel('x')
            plt.ylabel('y')

            U_ex = [u_ex(node) for node in nodes]

            plt.subplot(122)
            plt.plot(nodes[:,0], U_ex)
            plt.title("exact solution")
            plt.xlabel('x')
            plt.ylabel('y')

            # errors
            print "error:"
            print "\t|u - u_ex| = ", norm(U_ex-u)

    plt.show()

def plot_piecewise(u, nodes, u_ex=None, name=''):
    """ makes a piecewise plot of 1D solution
    Input:
        u - computed solution
        mesh - 1d mesh
        u_ex - exact solution (optional)
        name - name of figure """
    assert nodes.shape[1] == 1, 'Piecewise plotting only for 1 dim.'

    # set up figure
    if name == '':
        name = 'Plot piecewise constant in 1D'
    fig = plt.figure(num=name)

    nodes = mesh.nodes()
    n_nodes = mesh.n_nodes()

    x = np.linspace(0,1,101)
    vertices = np.unique(mesh.vertices())
    n_v = len(vertices)
    plt.plot(x, np.piecewise(x, [(x >= vertices[n]) & (x <= vertices[n+1]) for n in xrange(n_v-1)], u))

    # evaluate exact solution at nodes
    if u_ex is not None:
        U_ex = []
        for p in x:
            U_ex.append(u_ex([p]))

        plt.plot(x, U_ex)
        plt.legend(('computed','exact'))
    else:
        plt.legend('computed')

    plt.show()

if __name__ == '__main__':

    def piecewise_test(n=10):
        """ test piecewise plotting """
        from meshing import UnitIntMesh
        from functionSpace import LagrangeSpace

        u_ex = lambda x: 4*x[0]*(1.-x[0])     # exact solution
        mesh = UnitIntMesh(n)                 # 1d mesh
        fs = LagrangeSpace(mesh, 0)           # P0 function space

        # assemble
        A = fs.assemble()
        rhs = fs.rhs(u_ex)

        # solve with homogenous Dirichlet bc
        u = dirichlet_solve(mesh, A, rhs)
        plot_piecewise(u, mesh, u_ex)

    piecewise_test()


