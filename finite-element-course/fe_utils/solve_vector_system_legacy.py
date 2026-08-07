from fe_utils.finite_elements import LagrangeElement, VectorFiniteElement
from fe_utils.function_spaces import FunctionSpace, Function
from fe_utils.mesh import UnitSquareMesh
from fe_utils.quadrature import gauss_quadrature
from fe_utils.utils import errornorm
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import numpy as np


def assemble(fs, grad_f):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    #Element, degree and mesh from function space
    fe = fs.element
    degree = fe.degree
    mesh = fs.mesh

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2 * degree)

    # Tabulate the (local) basis functions and their gradients at the quadrature points.
    basis_tab = fe.tabulate(Q.points)
    grad_basis_tab = fe.tabulate(Q.points, grad=True) 

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes = fs.cell_nodes[c, :]
        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        J_inv_T = np.linalg.inv(J).T
        detJ = np.abs(np.linalg.det(J))

        #l
        #F evaluated
        F = grad_f.values[nodes]
        l[nodes] += np.einsum('j, qjc, qic,q->i', F, basis_tab, basis_tab, Q.weights) * detJ

        #A
        m = np.einsum('qjk, qik, q->ij', basis_tab, basis_tab, Q.weights) * detJ
        A[np.ix_(nodes, nodes)] += m


    return A, l

resolution = 2
degree = 2
mesh = UnitSquareMesh(resolution, resolution)
fe = LagrangeElement(mesh.cell, degree)
dimension = 2
fe = VectorFiniteElement(fe, dimension=dimension)
fs = FunctionSpace(mesh, fe)

f = Function(fs)
f.interpolate(lambda x: [x[i] for i in range(dimension)])

A, l = assemble(fs, f)

# Create the function to hold the solution.
u = Function(fs)

# Cast the matrix to a sparse format and use a sparse solver for
# the linear system. This is vastly faster than the dense
# alternative.
A = sp.csr_matrix(A)
u.values[:] = splinalg.spsolve(A, l)
print(f'Error norm: between u and grad_f is {errornorm(u, f)}')