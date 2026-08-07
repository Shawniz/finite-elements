import numpy as np
import sympy as sp
#from .reference_elements import ReferenceInterval, ReferenceTriangle
from fe_utils.reference_elements import ReferenceInterval, ReferenceTriangle
from itertools import product
np.seterr(invalid='ignore', divide='ignore')

def number_coordinates(degree, dimension):
    return (np.factorial(degree + dimension) / np.factorial(dimension)).astype(int)

def lagrange_points(cell, degree):
    """
    Construct the locations of the equispaced Lagrange nodes on a cell.
    :param cell: the ReferenceCell instance
    :param degree: the degree of polynomials for which to construct nodes.
    :returns: a rank 2 numpy array whose rows are the coordinates of the nodes.
    """
    topology, vertices = cell.topology, cell.vertices
    dim = vertices.shape[1]  # Assuming vertices is a numpy array
    coordinates_list = []

    for entity in topology:
        for entity_num in topology[entity]:
            if entity == 0:
                #Add vertices
                #print("-")
                print(vertices[topology[entity][entity_num]][0])
                coordinates = coordinates_list.append(vertices[topology[entity][entity_num][0]])
            elif entity == 1:
                #Add edges and in between
                start, end = vertices[topology[entity][entity_num]]
                coordinates_to_add = np.linspace(start, end, degree + 1)[1:-1]
                print(coordinates_to_add)
                if len(coordinates_to_add) > 0:
                    coordinates_list.append(coordinates_to_add)
                #print(entity_num)
                #new_edge_dict[entity_num] = coordinates_to_add
                #coordinate_count += len(coordinates_to_add)
            else:
                # Get vertices of the triangle
                print("LOL")
                #print(vertices[topology[entity][entity_num]])
                v0, v1, v2 = vertices[topology[entity][entity_num]]
                # Generate barycentric coordinates
                l1, l2 = np.meshgrid(np.linspace(0, 1, degree + 1), np.linspace(0, 1, degree + 1))
                l1 = l1.flatten()
                l2 = l2.flatten()
                l3 = 1 - l1 - l2
                increment = 1 / degree
                mask = (l3 > increment / 2) & (l1 > increment / 2) & (l2 > increment / 2)
                l1, l2, l3 = l1[mask], l2[mask], l3[mask]
                # Convert to Cartesian coordinates
                triangle_points = l1[:, np.newaxis] * v0 + l2[:, np.newaxis] * v1 + l3[:, np.newaxis] * v2
                print("Triang;e points")
                print(triangle_points)
                coordinates_list.append(triangle_points)  # Exclude vertices

    # Combine all points into a single array
    print(coordinates_list)
    return np.vstack(coordinates_list)

def generate_ntuples(n, p, current_tuple=()):
    """
    Generate all unique n-tuples of non-negative integers that sum to p.

    Args:
    n (int): The number of elements in each tuple.
    p (int): The total sum of the elements in each tuple.
    current_tuple (tuple): Used internally by the recursion to build up tuples.

    Returns:
    list of tuples: A list of all n-tuples that sum to p.
    """
    if n == 1:
        if p >= 0:  # Only include non-negative solutions
            return [(p,) + current_tuple ]
        else:
            return []
    else:
        result = []
        # Generate the first element, and recursively generate the rest
        for i in range(p + 1):  # i takes values from 0 to p
            result.extend(generate_ntuples(n - 1, p - i, current_tuple + (i,)))
        return result

def differentiate(expression, symbols, vars):
    # Compute the gradient (partial derivatives of f with respect to x and y)
    gradient_f = [sp.diff(expression, var) for var in vars]
    #print(gradient_f)
    return gradient_f

def create_symbolic_vars(dim):
    # Create a list of variable names based on dimension
    names = ['x' + str(i) for i in range(dim)]
    
    # Generate symbolic variables
    vars = sp.symbols(names)
    
    # Create dictionary to access variables by their string names
    var_dict = {str(v): v for v in vars}
    
    return vars, var_dict

def vandermonde_matrix(cell, degree, points, grad=False):
    """
    Construct the generalized Vandermonde matrix for polynomials of the specified degree
    on the cell provided.
    """
    num_points = points.shape[0]
    num_dimensions = points.shape[1]
    # Generate all combinations of powers up to 'degree' for 'num_dimensions'
    power_combs_all =  [comb for comb in product(range(degree + 1), repeat=num_dimensions) if sum(comb) <= degree]
    power_combs_all = sorted(power_combs_all, key=lambda x: (sum(x), -x[0]))

    print("Combs")
    print(power_combs_all)
    
    if not grad:
        # Calculate the size of the Vandermonde matrix
        num_terms = len(power_combs_all)
        vandermonde_matrix = np.zeros((num_points, num_terms))

        # Fill the Vandermonde matrix with the evaluated polynomials at each point
        for i, powers in enumerate(power_combs_all):
            term = np.prod([points[:, dim] ** power for dim, power in enumerate(powers)], axis=0)
            vandermonde_matrix[:, i] = term

        return vandermonde_matrix
    xs, ys = points[:,0], points[:,-1]
    dim = len(points[0,:])
    v = np.zeros((len(xs), 1, dim))
    print(v)
    #symbolically differentiate the terms
    print('points:', points)
    print('xs:', xs)
    for max_power in range(1, degree + 1):
        #power_combs = [(max_power - i, i) for i in range(max_power + 1)]
        #power_combs = generate_ntuples(len(points[0,:]), max_power)
        print("MAx power:", max_power)
        #print(power_combs)
        power_combs = [comb for comb in power_combs_all if sum(comb) == max_power]
        print(f'Dim:{dim}')
        variables, vars_dict = create_symbolic_vars(dim)
        print("variables:")
        print(variables)
        print(vars_dict)
        symbols = ' '.join([key for key in vars_dict]) 

        expressions = [np.prod(np.array(([s ** power for s, power in zip(variables, comb)]))) for comb in power_combs]
        print("expressions:")
        print(expressions)
        grads = np.array([[differentiate(expression, symbols, variables) for expression in expressions] for i in range(len(xs))])
        #subs[(sym, val) for sym, val in zip(variables, symbol_values)
        print(grads.shape)
        print(grads)

        #Lambdify this?
        for i in range(grads.shape[0]):
            for j in range(grads.shape[1]):
                grads[i,j] =  np.array([elem.subs([(var, coord) for var, coord in zip(variables, points[i,:])]) for elem in grads[i,j]])


        #grads = [[expr.subs([(sym, val) for sym, val in zip(variables, symbol_values)]) for expr in row] for row in grads]
        #grads = [[sp.derive_by_array(func, (x, y)) for func in row] for row in grads]
        print("Shape:")
        print(grads.shape)
        #grads = np.vstack((grads, grads, grads)).reshape(3, 2, -1)
        print(grads)
        print(np.transpose(grads, axes=[0,2,1]))
        print(grads.shape, v.shape)
        print(v)
        v = np.hstack((v, grads))
        print("v:", v.shape)
        print(v)
    print("-----")
    return v.astype(int)
    



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
        return np.einsum("ijk,jl->ilk", v, self.basis_coefs) if grad else np.matmul(v, self.basis_coefs)
      

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
        return [fn(v) for k,v in self.cell.topology[0]]

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
        super(LagrangeElement, self).__init__(cell, degree, nodes)
