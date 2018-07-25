import numpy as np
from functionSpace import FunctionSpace

class Dirichlet:
    """ class for implementing Dirichlet boundary conditions """
    def __init__(self, fs, g=0, *args):
        """
        assembles Dirichlet boundary condition, given by function (or constant) g on the boundary.
        parts of boundary can be specified according to names in boundary dictionary
        Input:
            fs - function space
            g - function (or constant), default is 0
            args: (optional)
                m (int) - if mixed space, gives number of element
                names (strings) - names for boundary dictionary
        """
        # check args input
        if len(args) == 0:
            m = 0; names = []
        elif isinstance(args[0], int):
            m = args[0]; names = [args[k] for k in range(1,len(args))]
        else:
            m = 0; names = args

        # get data from function space
        belt_to_nodes = fs.belt_to_nodes(m, *names)
        unique_nodes = np.unique(belt_to_nodes)
        node_to_coords = fs.node_to_coords(m)

        # assemble boundary condition
        bc = np.zeros(fs.n_nodes(m))
        if callable(g):
            bc[unique_nodes] = [g(node_to_coords[n]) for n in unique_nodes]
        else:
            bc[unique_nodes] = g

        #self.__belt_to_nodes = belt_to_nodes
        self.__unique_nodes = unique_nodes
        self.__bc = bc
        self.__m = m

    def m(self):
        """ return method number """
        return self.__m

    # def belt_to_nodes(self):
    #     """ return boundary element to node number matrix """
    #     return self.__belt_to_nodes

    def unique_nodes(self):
        """ return list of all node number involved in this bc """
        return self.__unique_nodes

    def assemble(self):
        """ assemble boundary condition """
        return self.__bc

