import numpy as np
import math

def plot_mesh(node_to_coords, elt_to_nodes, dim, dofs=None, file=None):
    """
    makes a plot of mesh defined by cell_coords and elt_to_nodes with or without node_to_coords
    Input:
        node_to_coords - node number to coordinates matrix
        elt_to_nodes - element number to node numbers matrix
        dim - spatial dimension
        dofs - DOF number to coordinate, if DOFs are shown in figure
        file - file name of figure is saved to .pdf file
    """
    import matplotlib.pyplot as plt

    if dofs is not None:
        plt.figure('Mesh plot with DOFs')
    else:
        plt.figure('Mesh plot')

    if dim == 2:
        # plot mesh
        x = node_to_coords[:,0]; y = node_to_coords[:,1]
        plt.gca().set_aspect('equal')
        plt.triplot(x, y, elt_to_nodes[:,:3], 'g-')

        # plot DOFs
        if dofs is not None:
            dof_x = dofs[:,0]; dof_y = dofs[:,1]
            plt.plot(dof_x,dof_y,'go')
            plt.title('Triangular mesh with DOFs.')
        else:
            plt.title('Triangular mesh.')

        plt.xlabel('x')
        plt.ylabel('y')
    else: # dim = 1
        # plot horizontal line
        xmin = np.min(node_to_coords); xmax = np.max(node_to_coords)
        plt.axhline(y=0, xmin=xmin, xmax=xmax, color='g', linewidth=2.)
        plt.ylim(-1,1)

        # add elt_to_nodes
        elt_to_vcoord = np.unique(node_to_coords[elt_to_nodes[:,:2]])
        for v in elt_to_vcoord:
            plt.axvline(v, color='k', linestyle='--', linewidth=0.5)
        plt.yticks([])

        if dofs is not None:
            y = np.zeros(len(dofs))
            plt.plot(dofs, y, 'go')
            plt.title('Interval mesh with DOFs.')
        else:
            plt.title('Interval mesh.')

    # save figure
    if file is not None:
        #plt.savefig(plotfile + '.png')
        plt.savefig(plotfile + '.pdf')

    plt.show()

#--------------------------------------------------------------------------------------#

def readGmsh(file):
    """
    Reads mesh data from .msh Gmsh-file
    File must specify at least two physical groups; boundary and interior
    Three groups can also be specified, e.g. Dirichlet and Neumann boundary node_to_coords
    Output:
        node_to_coords - node number to coordinate
        elt_to_nodes - element number to node numbers
        boundary - dictionary of other groups defined in mesh i.e. boundary
    """
    with open(file) as fp:
        print 'Read mesh file\n---------------------------------'
        print "Name:\n \t{}".format(fp.name)
        fp.readline() # '$MeshFormat\n'
        fp.readline() # version no.\n
        fp.readline() # '$EndMeshFormat\n'

        assert fp.readline() == '$PhysicalNames\n', 'File not read correctly.'
        n_phys_groups = int(fp.readline())

        names = []; dims = []
        print 'Physical groups:'
        for n in xrange(n_phys_groups):
            dim, num, name = fp.readline().split()
            print '\t{}, n={}, d={}'.format(name, num, dim)
            names.append(name[1:-1])
            dims.append(int(dim))
        assert fp.readline() == '$EndPhysicalNames\n', 'File not read correctly.'

        max_dim_name = names[dims.index(max(dims))] # name of group for elt_to_nodes of mesh

        # node_to_coords
        assert fp.readline() == '$Nodes\n', 'File not read correctly.'
        n_nodes = int(fp.readline())
        node_to_coords = np.fromfile(fp,dtype=float,count=n_nodes*4,sep=" ").reshape((n_nodes,4))
        node_to_coords = node_to_coords[:,1:3] # need only x and y coord
        assert len(node_to_coords) == n_nodes, 'Not all node_to_coords included.'
        assert fp.readline() == '$EndNodes\n', 'File not read correctly.'

        # elt_to_nodes
        assert fp.readline() == '$Elements\n', 'File not read correctly.'
        n_elts = int(fp.readline())

        # boundary groups
        boundary = {name : [] for name in names}
        for n in xrange(n_elts):
            words = fp.readline().split()
            group_tag = int(words[3])-1
            n_tags = int(words[2])
            local_nodes = [int(words[i])-1 for i in xrange(3+n_tags,len(words))]
            boundary[names[group_tag]].append(local_nodes)

        assert fp.readline() == '$EndElements\n', 'File not read correctly.'
        assert sum([len(boundary[name]) for name in names]) == n_elts, 'Not all elements included.'

        # extract group with highest dimension and delete from dictionary
        elt_to_nodes = np.array(boundary[max_dim_name], dtype=int)
        del boundary[max_dim_name]

    return node_to_coords, elt_to_nodes, boundary

#--------------------------------------------------------------------------------------#

def regular_mesh_data(box=[[0,1],[0,1]], res=[4,4], deg=1, diag='right'):
    """ assembles data for regular mesh (interval in 1D or triangular in 2D)
    Input:
        box - bounding box
        res - number of divisions of x (and y intervals)
        deg - degree of Langrange basis functions (1 through 3)
        diag - diagonal to the left or right (only for 2D data)
    Output:
        node_to_coords - node number to coordinate matrix
        elt_to_nodes - element number to node numbers matrix
    """

    # check valid data
    assert len(box) == len(res), 'Incompatible box and res arguments.'

    if len(box) == 1: # <-------------- 1D uniform interval mesh
        # data
        x = box[0]          # interval
        n = res[0]          # number of divisions of interval
        n_nodes = deg*n+1   # number of nodes

        # assemble node_to_coords
        temp = np.linspace(x[0],x[1],n_nodes)
        node_to_coords = []
        for node in temp:
            node_to_coords.append([node,0])
        node_to_coords = np.array(node_to_coords, dtype=float)

        # assemble elt_to_nodes
        elt_to_nodes = []
        for i in xrange(n):
            v0 = i*deg; v1 = v0+1; v2 = v1+1; v3 = v2+1
            if deg == 1:
                elt_to_nodes.append([v0, v1])
            elif deg ==2:
                elt_to_nodes.append([v0, v2, v1])
            elif deg == 3:
                elt_to_nodes.append([v0, v3, v1, v2])

        elt_to_nodes = np.array(elt_to_nodes, dtype=int)

    elif len(box) == 2: # <-------------- 2D uniform triangular mesh
        # check valid diagonal
        diag = diag.lower()
        assert diag == 'right' or diag == 'left', 'Invalid diagonal argument.'

        # data
        x = box[0]; y = box[1]                      # bounding box
        nx = res[0]; ny = res[1]                    # number of divisions of box
        nx_nodes = deg*nx+1; ny_nodes = deg*ny+1    # number of nodes in x and y directions
        n_nodes = nx_nodes*ny_nodes                 # total number of nodes
        n_elts = 2*nx*ny                            # number of elements

        # assemble node_to_coords
        xx = np.linspace(x[0],x[1],nx_nodes)
        yy = np.linspace(y[0],y[1],ny_nodes)
        node_to_coords = []
        for iy in xrange(nx_nodes):
            for ix in xrange(ny_nodes):
                node_to_coords.append([xx[ix], yy[iy]])
        node_to_coords = np.array(node_to_coords, dtype=float)
        assert len(node_to_coords) == n_nodes, 'Assembly of node_to_coords failed.'

        # assemble elt_to_nodes, anti-clockwise numbering of nodes
        elt_to_nodes = []
        if deg == 1:
            for iy in xrange(ny):
                for ix in xrange(nx):
                    v0 = iy*nx_nodes+ix; v1 = v0+1
                    v2 = v0+nx_nodes; v3 = v2+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v1, v3])
                        elt_to_nodes.append([v0, v3, v2])
                    else:
                        elt_to_nodes.append([v0, v1, v2])
                        elt_to_nodes.append([v1, v3, v2])
        elif deg == 2:
            for iy in xrange(ny):
                for ix in xrange(nx):
                    v0 = (2*iy)*nx_nodes+(2*ix); v1 = v0+1; v2 = v1+1
                    v3 = v0+nx_nodes; v4 = v3+1; v5 = v4+1
                    v6 = v3+nx_nodes; v7 = v6+1; v8 = v7+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v2, v8, v1, v5, v4])
                        elt_to_nodes.append([v0, v8, v6, v4, v7, v3])
                    else:
                        elt_to_nodes.append([v0, v2, v6, v1, v4, v3])
                        elt_to_nodes.append([v2, v8, v6, v5, v7, v4])
        elif deg == 3:
            for iy in xrange(ny):
                for ix in xrange(nx):
                    v0 = (3*iy)*nx_nodes+(3*ix); v1 = v0+1; v2 = v1+1; v3 = v2+1
                    v4 = v0+nx_nodes; v5 = v4+1; v6=v5+1; v7=v6+1
                    v8 = v4+nx_nodes; v9 = v8+1; v10 = v9+1; v11 = v10+1
                    v12 = v8+nx_nodes; v13 = v12+1; v14 = v13+1; v15 = v14+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v3, v15, v1, v2, v7, v11, v10, v5, v6])
                        elt_to_nodes.append([v0, v15, v12, v5, v10, v14, v13, v8, v4, v9])
                    else:
                        elt_to_nodes.append([v0, v3, v12, v1, v2, v6, v9, v8, v4, v5])
                        elt_to_nodes.append([v3, v15, v12, v7, v11, v14, v13, v9, v6, v10])


        elt_to_nodes = np.array(elt_to_nodes, dtype=int)
        assert len(elt_to_nodes) == n_elts, 'Assembly of elt_to_nodes failed.'

    else:
        raise ValueError('Invalid box and res arguments.')

    return node_to_coords, elt_to_nodes

#--------------------------------------------------------------------------------------#

def regular_mesh_bdata(res=[4,4], deg=1):
    """ assembles boundary data for regular mesh (interval in 1D or triangular in 2D)
    Input:
        res - number of divisions of x (and y intervals)
        deg - degree of Langrange basis functions (1 through 3)
    Output:
        boundary - dictionary of edge number to node numbers matrices
    """

    # assemble boundary
    if len(res) == 1: # <-------------- 1D uniform interval mesh
        n = res[0]                                  # number of divisions of interval
        n_nodes = deg*n+1                           # number of node_to_coords

        boundary = {'left': [[0]], 'right': [[n_nodes-1]]}

    elif len(res) == 2: # <-------------- 2D uniform triangular mesh
        nx = res[0]; ny = res[1]                    # number of divisions of interval
        n_bedges = 2*nx + 2*ny                      # number of boundary edges
        nx_nodes = deg*nx+1; ny_nodes = deg*ny+1    # number of nodes in x and y directions

        bottom = []; top = []; right = []; left = []
        # assemble edge_to_nodes
        for n in xrange(nx):
            n *= deg
            bnodes = [n+i for i in xrange(deg+1)]
            bottom.append([bnodes[0], bnodes[-1]] + [bnodes[i] for i in xrange(1,deg)])

            tnodes = [nx_nodes*ny_nodes - (n+i) for i in xrange(1,deg+2)]
            top.append([tnodes[0], tnodes[-1]] + [tnodes[i] for i in xrange(1,deg)])

        for n in xrange(ny):
            rnodes = [(n*deg+i)*nx_nodes-1 for i in xrange(1,deg+2)]
            right.append([rnodes[0], rnodes[-1]] + [rnodes[i] for i in xrange(1,len(rnodes)-1)])

            lnodes = [(n*deg+i)*nx_nodes for i in xrange(deg+1)]
            left.append([lnodes[-1], lnodes[0]] + [lnodes[i] for i in xrange(1,len(lnodes)-1)])

        left.reverse()
        boundary = {'bottom':bottom, 'top':top, 'right':right, 'left':left}

    else:
        raise ValueError('Invalid res argument.')

    return boundary

#--------------------------------------------------------------------------------------#

class SuperMesh(object):
    """ super class for meshes """
    def __init__(self, node_to_coords, elt_to_nodes, boundary, dim):
        """
        Input:
            node_to_coords - node number to coordinate
            elt_to_nodes - elt number to node numbers
            boundary - dictionary of boundary groups (edge to nodes)
            dim - spatial dimension of mesh
        """
        # assemble elt number to vertex coords
        elt_to_vcoords = node_to_coords[elt_to_nodes[:,:dim+1]]

        # if 2D assemble boundary vertices
        if dim == 2:
            bvertices = {}
            for name in boundary.keys():
                edge_to_nodes = np.array(boundary[name], dtype=int)
                bvertices[name] = np.array([node_to_coords[e[:2]] for e in edge_to_nodes], dtype=float)
        else:
            bvertices = None

        self.__dim = dim
        self.__node_to_coords = node_to_coords
        self.__elt_to_nodes = elt_to_nodes
        self.__n_elts = elt_to_nodes.shape[0]
        self.__elt_to_vcoords = elt_to_vcoords
        self.__boundary = boundary
        self.__bvertices = bvertices

    def dim(self):
        """ return spatial dimension of mesh """
        return self.__dim

    def n_elts(self):
        """ return number of elements in mesh """
        return self.__n_elts

    def elt_to_vcoords(self):
        """ return element number to vertex coords matrix """
        return self.__elt_to_vcoords

    def elt_to_ccoords(self, n=None):
        """ return element number to center coord matrix """
        return np.array([sum(v)/(self.dim()+1) for v in self.__elt_to_vcoords], dtype=float)

    # def get_bkeys(self):
    #     """ return all keys in boundary dictionary """
    #     return self.__boundary.keys()

    def bvertices(self):
        """ return edge number to vertex coord matrix dictionary """
        if self.__bvertices is None:
            raise ValueError('Edge vertex coords only available in 2D.')
        else:
            return self.__bvertices

    def plot(self, dofs=None, file=None):
        """ plot figure of mesh
        DOFs given by dofs can be shown in figure
        file - name of .pdf file if figure is saved
        """
        plot_mesh(self.__node_to_coords, self.__elt_to_nodes, self.dim(), dofs=dofs, file=file)

#--------------------------------------------------------------------------------------#

class Gmesh(SuperMesh):
    """ 2D mesh constructed from .msh Gmsh-file"""
    def __init__(self, file):
        """
        Input:
            file - .msh Gmsh-file defining node_to_coords, elt_to_nodes and boundary
        """
        node_to_coords, elt_to_nodes, boundary = readGmsh(file)
        dim = 2
        n_elt_nodes = elt_to_nodes.shape[1]
        super(Gmesh, self).__init__(node_to_coords, elt_to_nodes, boundary, dim)
        self.__deg = int( (math.sqrt(9-8*(1-n_elt_nodes))-3)/2 )

    def deg(self):
        """ return degree of Lagrange basis functions """
        return self.__deg

    def get_data(self):
        """ get mesh data
        Returns:
            node_to_coords - coords of node_to_coords
            elt_to_nodes - interior elt_to_nodes of mesh
            boundary - boundary group dictionary
        """
        return self._SuperMesh__node_to_coords, self._SuperMesh__elt_to_nodes

    def get_bdata(self):
        """ get boundary data """
        return self.__boundary

#--------------------------------------------------------------------------------------#

class RegularMesh(SuperMesh):
    """ super class for regular meshes """
    def __init__(self, box=[[0,1], [0,1]], res=[4,4], diag='right'):
        """
        Input: (if 1D, then first arg of box and res neglected)
            box - bounding box
            res - subdivisions of box in x (and y directions)
            diag - left or right (only for 2D mesh)
        """
        # get data
        node_to_coords, elt_to_nodes = regular_mesh_data(box=box, res=res, diag=diag)
        boundary = regular_mesh_bdata(res=res)

        dim = len(box)
        super(RegularMesh, self).__init__(node_to_coords, elt_to_nodes, boundary, dim)
        self.__box = box
        self.__res = res
        self.__diag = diag

    def mesh_size(self):
        """ return h - size of elements in mesh """
        if self.dim() == 2:
            return max((self.__box[0][1] - self.__box[0][0])/float(self.__res[0]),\
                       (self.__box[1][1] - self.__box[1][0])/float(self.__res[1]))
        else:
            return (self.__box[0][1] - self.__box[0][0])/float(self.__res[0])

    def get_data(self, deg=1):
        """ get mesh data
        Input:
            deg - degree of Lagrange basis functions
        Returns:
            node_to_coords - node number to coordinate matrix
            elt_to_nodes - elt number to node numbers matrix
        """

        # check valid degree
        assert isinstance(deg, int) and 0 <= deg and deg <= 3, 'Invalid degree.'

        # get data
        if deg == 0:
            node_to_coords = self.elt_to_ccoords()
            elt_to_nodes = np.array([[n] for n in xrange(self.n_elts())])
        elif deg == 1:
            node_to_coords = self._SuperMesh__node_to_coords
            elt_to_nodes = self._SuperMesh__elt_to_nodes
        else: # deg is 2 or 3
            node_to_coords, elt_to_nodes = regular_mesh_data(self.__box, self.__res, deg=deg, diag=self.__diag)
        return node_to_coords, elt_to_nodes

    def get_bdata(self, deg=1):
        """ get mesh boundary data
        Input:
            deg - degree of Lagrange basis functions
        Returns:
            boundary - dictionary of edge_to_nodes """

        # check valid degree
        assert isinstance(deg, int) and 0 <= deg and deg <= 3, 'Invalid degree.'

        # get boundary data
        if deg == 0:
            if self.dim() == 2:
                x,y = self.__box; nx,ny = self.__res

                # assemble boundary groups
                bottom = []; top = []; left = []; right = []
                for n in xrange(nx):
                    bottom.append([2*n])
                    top.append([2*(nx*ny-n)-1])

                for n in xrange(ny):
                    left.insert(0,[2*n*nx+1])
                    right.append([2*(nx*(n+1)-1)])

                return {'bottom':bottom, 'top':top, 'right':right, 'left':left}
            else: # dim = 1
                # assemble boundary groups
                return {'left':[[0]], 'right':[[self.n_elts()-1]]}
        elif deg == 1:
            return self._SuperMesh__boundary
        else: # 2 or 3
            return regular_mesh_bdata(self.__res, deg)

#--------------------------------------------------------------------------------------#

class UnitSquareMesh(RegularMesh):
    """ unit square mesh """
    def __init__(self, nx=4, ny=4, diag='right'):
        """
        Input:
            nx,ny - subdivisions in x and y directions
            diag - left or right
        """
        super(UnitSquareMesh, self).__init__(box=[[0,1],[0,1]], res=[nx,ny], diag=diag)

#--------------------------------------------------------------------------------------#

class UnitIntMesh(RegularMesh):
    """ unit interval mesh """
    def __init__(self, n=10):
        """
        Input:
            n - number of divisions of unit interval
        """
        super(UnitIntMesh, self).__init__(box=[[0,1]], res=[n])


if __name__ == '__main__':

    # mesh
    mesh = UnitSquareMesh(2,2)
    #mesh = UnitIntMesh(4)
    #mesh = Gmesh('Meshes/square_P2.msh')

    # mesh data
    node_to_coords, elt_to_nodes = mesh.get_data()

    #mesh.plot(node_to_coords) <----------------- OK for all meshes with DOFs

    # mesh bdata
    boundary = mesh.get_bdata()
    print boundary['left']




