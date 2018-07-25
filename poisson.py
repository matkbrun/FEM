import numpy as np
import math
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve, cg
from meshing import RegularMesh, UnitSquareMesh, UnitIntMesh
from functionSpace import FunctionSpace
from finiteElement import FiniteElement
from boundaryCond import Dirichlet
from aux import plot_sol, fem_solver


"""
#--------------------------------------------------------------------------------------#

Different examples solving the Poisson equation
with Dirichlet/Neuman boundary conditions

    - Laplace(u) = f,       in Omega,
               u = u_D,     on Gamma_D,
  Nabla(u) dot n = u_N,     on Gamma_N,

    where Gamma_D U Gamma_N = partial Omega

#--------------------------------------------------------------------------------------#
"""

def dirichlet_ex1(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ homogenous Dirichlet bc in 2D
    - P1 conv rate: ~2
    - P2 conv rate: ~8
    - P3 conv rate: ~.5 (not working) """

    print "\nDirichlet ex1 in 2D"
    # data
    mesh = UnitSquareMesh(n,n,diag=diag)
    f = lambda x: 32.*(x[1]*(1.-x[1]) + x[0]*(1.-x[0]))
    u_ex = lambda x: 16.*x[0]*(1.-x[0])*x[1]*(1.-x[1])

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    # solve w/ homogenous Dirichlet bc
    bc = Dirichlet(fs)
    u = fem_solver(fs, A, rhs, bc)

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def dirichlet_ex2(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ Dirichlet bc in 2D
    - P1 conv rate: ~2
    - P2 conv rate: ~8
    - P3 conv rate: ~.5 (not working) """

    print "\nDirichlet ex2 in 2D"
    # data
    mesh = UnitSquareMesh(n,n,diag=diag)
    f = lambda x: -6*x[0]*x[1]*(1.-x[1]) + 2*pow(x[0],3)
    g = lambda x: x[1]*(1.-x[1])
    u_ex = lambda x: x[1]*(1.-x[1])*pow(x[0],3)

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    bc1 = Dirichlet(fs, 0, 'bottom', 'top', 'left')
    bc2 = Dirichlet(fs, g, 'right')

    # solve
    u = fem_solver(fs, A, rhs, [bc1, bc2])

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def dirichlet_ex3(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ Dirichlet bc in 2D
    - P1 conv rate: ~2
    - P2 conv rate: ~7
    - P3 conv rate: ~.5 (not working) """

    print "\nDirichlet ex3 in 2D"
    # data
    mesh = RegularMesh(box=[[0,math.pi], [0,math.pi]], res=[n,n], diag=diag)
    f = 0
    u_ex = lambda x: math.sinh(x[0])*math.cos(x[1])
    g_left = 0
    g_right = lambda x: math.sinh(math.pi)*math.cos(x[1])
    g_bottom = lambda x: math.sinh(x[0])
    g_top = lambda x: -math.sinh(x[0])

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    bc_left = Dirichlet(fs, g_left, 'left')
    bc_right = Dirichlet(fs, g_right, 'right')
    bc_bottom = Dirichlet(fs, g_bottom, 'bottom')
    bc_top = Dirichlet(fs, g_top, 'top')

    # solve
    u = fem_solver(fs, A, rhs, [bc_left, bc_right, bc_bottom, bc_top])

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def dirichlet_ex4(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ Dirichlet bc in 2D
    - P1 conv rate: ~2
    - P2 conv rate: ~8
    - P3 conv rate: ~.5 (not working) """

    print "\nDirichlet ex4 in 2D"
    # data
    mesh = RegularMesh(box=[[0,math.pi], [0,math.pi]], res=[n,n], diag=diag)
    f = 0
    u_ex = lambda x: math.cosh(x[0])*math.sin(x[1])
    g_left = lambda x: math.sin(x[1])
    g_right = lambda x: math.cosh(math.pi)*math.sin(x[1])
    g_bottom = 0
    g_top = 0

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    bc_left = Dirichlet(fs, g_left, 'left')
    bc_right = Dirichlet(fs, g_right, 'right')
    bc_bottom = Dirichlet(fs, g_bottom, 'bottom')
    bc_top = Dirichlet(fs, g_top, 'top')

    # solve
    u = fem_solver(fs, A, rhs, [bc_left, bc_right, bc_bottom, bc_top])

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def dirichlet_ex5(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ Dirichlet bc in 1D
    seems to work for P1, P2 and P3, but
    not correct conv rates due to very small errors """

    print "\nDirichlet ex5 in 1D"
    # data
    mesh = UnitIntMesh(n)
    f = 8.
    u_ex = lambda x: 4.*x[0]*(1.-x[0])

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    # solve
    bc = Dirichlet(fs)
    u = fem_solver(fs, A, rhs, bc)

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def neuman_ex1(n=16, deg=1, plot=True, gauss=4, diag='right'):
    """ Poisson w/ Neuman bc in 2D
    ill conditioned -> use CG
    - P1 conv rate: ~2
    - P2 conv rate: ~6
    - P3 conv rate: ~.5 (not working)"""

    print "\nNeuman ex1 in 2D"
    # data
    mesh = UnitSquareMesh(n,n,diag=diag)
    c = 2*math.pi
    f = lambda x: 2*pow(c,2)*math.sin(c*(x[0] + x[1]))
    u_ex = lambda x: math.sin(c*(x[0] + x[1]))

    g_bottom = lambda x: -c*math.cos(c*x[0])
    g_right = lambda x: c*math.cos(c*(1. + x[1]))
    g_top = lambda x: c*math.cos(c*(1. + x[0]))
    g_left = lambda x: -c*math.cos(c*x[1])

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    bottom = fs.assemble_boundary(g_bottom, 'bottom')
    right = fs.assemble_boundary(g_right, 'right')
    top = fs.assemble_boundary(g_top, 'top')
    left = fs.assemble_boundary(g_left, 'left')

    rhs += bottom + right + top + left

    #solve
    u = fem_solver(fs, A, rhs, CG=True)
    # u, info = cg(A, rhs)
    # if info == 0:
    #     print "CG convergence OK"
    # else:
    #     print "failed to converge"

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def neuman_ex2(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ homogenous Neuman bc in 1D
    ill conditioned -> use CG
    - P1 conv rate: ~1.5
    - P2 conv rate: ~1.5
    - P3 conv rate: ~1.5 """

    print "\nNeuman ex2 in 1D"
    # data
    mesh = UnitIntMesh(n)
    f = lambda x: pow(2*math.pi, 2)*math.cos(2*math.pi*x[0])
    u_ex = lambda x: math.cos(2*math.pi*x[0])

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    # solve
    u = fem_solver(fs, A, rhs, CG=True)
    # u, info = cg(A, rhs)
    # if info == 0:
    #     print "CG convergence OK"
    # else:
    #     print "failed to converge"

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def neuman_ex3(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ homogenous Neuman bc in 2D
    ill conditioned -> use CG
    - P1 conv rate: ~2.3
    - P2 conv rate: ~6
    - P3 conv rate: ~.5 (not working)"""

    print "\nNeuman ex3 in 2D"
    # data
    mesh = UnitSquareMesh(n,n,diag=diag)
    pi = math.pi
    f = lambda x: pi*(math.cos(pi*x[0]) + math.cos(pi*x[1]))
    u_ex = lambda x: (math.cos(pi*x[0]) + math.cos(pi*x[1]))/pi

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    # solve
    u = fem_solver(fs, A, rhs, CG=True)
    # u, info = cg(A,rhs)
    # if info == 0:
    #     print "CG convergence OK"
    # else:
    #     print "failed to converge"

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)
    U_ex = [u_ex(node) for node in nodes]

    # return error and mesh size
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def neuman_ex4(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ Neuman bc in 2D
    ill conditioned -> use CG
    - P1 conv rate: ~1.5
    - P2 conv rate: ~1.1
    - P3 conv rate: ~.5 (not working) """

    print "\nNeuman ex4 in 2D"
    # data
    mesh = RegularMesh(box=[[0,math.pi], [0,math.pi]], res=[n,n], diag=diag)
    f = 0
    u_ex = lambda x: math.cos(2*x[0])*math.cosh(2*x[1])/200
    g_top = lambda x: 2*math.cos(2*x[0])*math.sinh(2*math.pi)/200

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    top = fs.assemble_boundary(g_top, 'top')
    rhs += top

    # solve
    u = fem_solver(fs, A, rhs, CG=True)
    #u = spsolve(A, rhs)
    # u, info = cg(A,rhs)
    # if info == 0:
    #     print "CG convergence OK"
    # else:
    #     print "failed to converge"

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def mixed_ex1(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ mixed bc in 1D
    seems to work for P1, P2 and P3, but
    not correct conv rates due to very small errors """

    print "\nMixed bc example in 1D"
    # data
    mesh = UnitIntMesh(n)
    d = 1; c = 1;
    u_ex = lambda x: 1. - x[0]**2 + d + c*(x[0]-1.)
    f = 2

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)
    rhs[0] -= c                             # Neuman on left boundary
    right_bc = Dirichlet(fs, d, 'right')    # Dirichlet on right boundary

    # solve
    u = fem_solver(fs, A, rhs, right_bc)

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)

    # return error and mesh size
    U_ex = [u_ex(node) for node in nodes]
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def mixed_ex2(n=16, deg=1, plot=True, gauss=4, diag='right'): # <---------------------------- works
    """ Poisson w/ mixed bc in 2D
    P1 conv rate: ~2.1
    P2 conv rate: ~6
    P3 conv rate: ~.5 (not working) """

    print "\nMixed bc example in 2D"
    # data
    mesh = UnitSquareMesh(n,n,diag=diag)
    f = lambda x: 2*(2*pow(x[1],3) - 3*pow(x[1],2) + 1.) - 6*(1.-pow(x[0],2))*(2*x[1]-1.)
    g = lambda x: 2*pow(x[1],3) - 3*pow(x[1],2) + 1.
    u_ex = lambda x: (1.-pow(x[0],2))*(2*pow(x[1],3) - 3*pow(x[1],2) + 1.)

    # assemble
    fs = FunctionSpace(mesh, FiniteElement('lagrange', deg), gauss)
    A = fs.stiffness()
    rhs = fs.rhs(f)

    bc1 = Dirichlet(fs, 0, 'right')
    bc2 = Dirichlet(fs, g, 'left')

    # solve (homogenous Neuman on top, bottom)
    u = fem_solver(fs, A, rhs, [bc1, bc2])

    nodes = fs.node_to_coords()
    if plot:
        plot_sol(u,nodes,u_ex)
    U_ex = [u_ex(node) for node in nodes]

    # return error and mesh size
    return norm(U_ex - u), mesh.mesh_size()

#--------------------------------------------------------------------------------------#

def conv_test(dim, ex, deg=1, gauss=4, diag='right'):
    import matplotlib.pyplot as plt
    """ convergence test for above examples of Poisson equation
        Input:
            ex - Poisson example
            deg - degree of Lagrange function space
            gauss - degree of Gaussian quadrature
            diag - left ot right
        Output:
            makes a plot of errors vs mesh sizes, also plots h and h^2
    """
    # mesh resulotions
    if dim == 1:
        res = [4, 8, 16, 32, 64]
    else: # dim = 2
        res = [2, 4, 8, 16, 32]

    # mesh sizes and errors
    mesh_sizes = []; errors = []

    for n in res:
        e, h = ex(n,deg,False,gauss,diag)
        errors.append(e); mesh_sizes.append(h)

    print "-------------------------------------------------------"
    print "Poisson convergence test completed."
    print "\tmesh sizes: ", mesh_sizes
    print "\terrors: ", errors
    print "\nratios: e_n/e_{n+1}"
    for j in range(len(errors)-1):
        print "\t", errors[j]/errors[j+1]
    print "-------------------------------------------------------"

    plt.figure('Poisson convergence test')
    plt.loglog(mesh_sizes,errors)
    plt.loglog(mesh_sizes,mesh_sizes)
    plt.loglog(mesh_sizes,np.array(mesh_sizes)**2)
    plt.legend(('error', 'h', 'h^2'))
    plt.xlabel('h')
    plt.ylabel('error')
    plt.show()


if __name__ == '__main__':

    ##e,h = dirichlet_ex1(n=16,deg=1) # 2D
    ##e,h = dirichlet_ex2(n=16,deg=1) # 2D
    ##e,h = dirichlet_ex3(n=16,deg=1) # 2D
    ##e,h = dirichlet_ex4(n=16,deg=1) # 2D
    ##e,h = dirichlet_ex5(n=16,deg=1) # 1D

    ##e,h = neuman_ex1(n=16,deg=1) # 2D
    ##e,h = neuman_ex2(n=16,deg=3) # 1D
    #e,h = neuman_ex3(n=16,deg=1) # 2D
    #e,h = neuman_ex4(n=16,deg=1) # 2D

    #e,h = mixed_ex1(n=16,deg=1) # 1D
    #e,h = mixed_ex2(n=16,deg=1) # 2D
blablamslabla
    dim = 2
    conv_test(dim, neuman_ex1, deg=1, gauss=4, diag='right')












