import numpy as np
import sympy as sp
import math
#from .reference_elements import ReferenceInterval, ReferenceTriangle
from fe_utils.reference_elements import ReferenceInterval, ReferenceTriangle
from itertools import product, chain
import copy
np.seterr(invalid='ignore', divide='ignore')

def lagrange_points(cell, degree):
    """
    Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.
    """
    topology, vertices = cell.topology.copy(), cell.vertices
    topology.pop(0)
    dim = vertices.shape[1]  # Assuming vertices is a numpy array
    coordinates_list = vertices

    for entity in topology:
        for entity_num in topology[entity]:
            pos = topology[entity][entity_num]
            points = [vertices[i] for i in pos]
            generated_points = generate_points(points, degree)
            if len(generated_points) > 0:
                coordinates_list = np.vstack((coordinates_list, generated_points))
    # Combine all points into a single array
    return coordinates_list

def linear_combination(vertices, comb):
    return np.dot(np.array(comb), vertices)

def generate_points(vertices, degree):
    num_vertices = len(vertices)
    resolution = 1 / degree

    # This grid will generate coordinates for the simplex defined by the vertices
    #from 1 to degree - 1 so we don't generate edges or vertices

    power_combs_all =  [comb for comb in product(range(1, degree), repeat=num_vertices) if sum(comb) == degree]
    power_combs_all = np.array(sorted(power_combs_all, key=lambda x: (sum(x), -x[0])))
    points = [linear_combination(vertices, comb) * resolution for comb in power_combs_all]
    
    return points

def differentiate(expression, symbols, vars):
    # Compute the gradient (partial derivatives of f with respect to x and y)
    gradient_f = [sp.diff(expression, var) for var in vars]
    return gradient_f

def create_symbolic_vars(dim):
    # Create a list of variable names based on dimension
    names = ['x' + str(i) for i in range(dim)]
    
    # Generate symbolic variables
    variables = sp.symbols(names)
    
    # Create dictionary to access variables by their string names
    var_dict = {str(v): v for v in variables}
    return variables, var_dict

def vandermonde_matrix(cell, degree, points, grad=False):
    """
    Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    points = np.asarray(points)
    num_points = points.shape[0]
    num_dimensions = points.shape[1]
    # Generate all combinations of powers up to 'degree' for 'num_dimensions'
    power_combs_all =  [comb for comb in product(range(degree + 1), repeat=num_dimensions) if sum(comb) <= degree]
    power_combs_all = sorted(power_combs_all, key=lambda x: (sum(x), -x[0]))
    
    if not grad:
        # Calculate the size of the Vandermonde matrix
        num_terms = len(power_combs_all)
        vandermonde_matrix = np.zeros((num_points, num_terms))

        # Fill the Vandermonde matrix with the evaluated polynomials at each point
        for i, powers in enumerate(power_combs_all):
            term = np.prod([points[:, dim] ** power for dim, power in enumerate(powers)], axis=0)
            vandermonde_matrix[:, i] = term

        return vandermonde_matrix
    
    xs = points[:,0]
    dim = len(points[0,:])
    v = np.zeros((len(xs), 1, dim))

    #Symbolically differentiate the terms
    for max_power in range(1, degree + 1):
        power_combs = [comb for comb in power_combs_all if sum(comb) == max_power]
        variables, vars_dict = create_symbolic_vars(dim)
        symbols = ' '.join([key for key in vars_dict]) 

        #Expressions and their derivatives
        expressions = [np.prod(np.array(([s ** power for s, power in zip(variables, comb)]))) for comb in power_combs]
        grads = np.array([[differentiate(expression, symbols, variables) for expression in expressions] for i in range(len(xs))])
        # Substitue values to evaluate the grad #Lambdify this?
        for i in range(grads.shape[0]):
            for j in range(grads.shape[1]):
                grads[i,j] =  np.array([elem.subs([(var, coord) for var, coord in zip(variables, points[i,:])]) for elem in grads[i,j]])

        v = np.hstack((v, grads.astype(float)))
    return v
    


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))
        # to an array of polynomial coefficients defining the basis functions.

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        v = vandermonde_matrix(self.cell, self.degree, points, grad)
        return np.einsum("ij...,jl->il...", v, self.basis_coefs) 
      

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return [fn(node) for node in nodes]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        topology = cell.topology
        dimension = cell.dim
        count = 0
        entity_nodes = {}
        for entity in topology:
            entity_nodes[entity] = {}
            for obj in topology[entity]:
                rng = int(math.factorial(degree - 1) / (math.factorial(entity) * math.factorial(degree - 1 - entity))) if degree - 1 - entity >= 0 else 0
                if rng == 0:
                    entity_nodes[entity][0] = []
                    break
                entity_nodes[entity][obj] = [count + i for i in range(rng)]
                count = entity_nodes[entity][obj][-1] + 1
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)

class VectorFiniteElement(object):
    def __init__(self, fe, dimension=2):
        """A finite element defined over cell.

        :param fe: the :class:`~.finite_elements.FiniteElement`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """
        #Finite element whcih we are making vector
        self.fe = fe

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = fe.cell

        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = fe.degree

        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = fe.nodes

        #Dimension of the finite element
        self.dimension = dimension

        #: The number of nodes in this element.
        self.node_count = self.nodes.shape[0] * self.dimension

        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)` for the non-vector finite element.
        self.entity_nodes = fe.entity_nodes

        if self.entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(fe.entity_nodes[d][0]) * dimension
                                              for d in range(self.cell.dim + 1)])


        self.entity_nodes = {entity: {entity_num: list(chain(*[range(n * dimension, (n + 1) * dimension) for n in node_num])) 
                                for entity_num, node_num in outer_val.items()}
                                    for entity, outer_val in self.entity_nodes.items()}

        self.basis_coefs = self.tabulate(fe.nodes)

        #Weight matrix for interpolation
        self.node_weights = np.eye(dimension)

        #Now make correct self.nodes
        nodes = list(chain(*[[node] * dimension for node in self.nodes]))
        self.nodes = nodes
    
    def tabulate(self, points, grad=False):
        tabulate = self.fe.tabulate(points, grad=grad)
        tab_shape = (tabulate.shape[0], self.dimension * tabulate.shape[1], tabulate.shape[2], self.dimension) if grad \
                        else (tabulate.shape[0], self.dimension * tabulate.shape[1], self.dimension)
        output = np.zeros(tab_shape)

        for d in range(self.dimension):
            #Every second column put tabulate into the respective column
            if grad:
                output[:, d::self.dimension, :, d] = tabulate
            else:
                output[:, d::self.dimension, d] = tabulate

        return output
      
    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a vector value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return [np.dot(fn(node), self.node_weights[i % self.dimension]) for i, node in enumerate(nodes)]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)

