"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from argparse import ArgumentParser
from fe_utils.finite_elements import LagrangeElement, VectorFiniteElement
from fe_utils.function_spaces import FunctionSpace, Function
from fe_utils.mesh import UnitSquareMesh
from fe_utils.quadrature import gauss_quadrature
from fe_utils.utils import errornorm
import numpy as np
from numpy import cos, pi, sin
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def assemble(fs_u, fs_p, f_u):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    #Element, degree and mesh from function space V
    fe_u = fs_u.element
    degree_u = fe_u.degree

    #Element, degree and mesh from function space Q
    fe_p = fs_p.element

    #Share mesh
    mesh = fs_u.mesh

    # Create an appropriate (complete) quadrature rule. Degree is larger for the u-element.
    Q = gauss_quadrature(fe_u.cell, 2 * degree_u)

    # Tabulate the (local) basis functions and their gradients at the quadrature points for u and p
    basis_tab_u = fe_u.tabulate(Q.points)
    grad_basis_tab_u = fe_u.tabulate(Q.points, grad=True)

    basis_tab_p = fe_p.tabulate(Q.points)
    grad_basis_tab_p = fe_p.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs_u.node_count, fs_u.node_count))
    B = sp.lil_matrix((fs_p.node_count, fs_u.node_count))
    l = np.zeros(fs_u.node_count + fs_p.node_count)

    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes_u = fs_u.cell_nodes[c, :]
        nodes_p = fs_p.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        J_inv_T = np.linalg.inv(J).T
        detJ = np.abs(np.linalg.det(J))

        #F evaluated
        F_u = f_u.values[nodes_u]

        #l
        l[nodes_u] += np.einsum('j, qjc, qic,q->i', F_u, basis_tab_u, basis_tab_u, Q.weights) * detJ

        #A
        eps_term = np.einsum('cd, abdk ->abck', J_inv_T, grad_basis_tab_u)
        eps_local = eps_term + np.transpose(eps_term, (0, 1, 3, 2))
        eps = np.einsum('q, qjab, qiab -> ij', Q.weights, eps_local, eps_local)
        A[np.ix_(nodes_u, nodes_u)] += detJ/4 * eps #* 2

        #B
        trace_like = np.einsum('ijcc->ij', eps_term)
        B[np.ix_(nodes_p, nodes_u)] += -detJ * np.einsum("q, qi, qj -> ij", Q.weights, basis_tab_p, trace_like)
    
    #Stitch the sparse matrix
    blocks = [[A, B.T],[B, None]]
    M = sp.bmat(blocks, format='lil')

    #Apply boundary conditions to A and l
    bn_nodes_u = boundary_nodes(fs_u)
    M[bn_nodes_u, :] = 0
    M[bn_nodes_u, bn_nodes_u] = 1  
    l[bn_nodes_u] = 0
    
    #Apply boundary conditions to B, zero out first basis function
    M[fs_u.node_count, :] = 0
    M[:, fs_u.node_count] = 0

    #Not needed as already zero
    #l[fs_u.node_count] = 0

    #We have a full zero row and zero column. Want to remove nonsingularity and don't want to affect calulations. 
    #So, place zero in first element of B. That removes full-zero row and column and doesn't affect computation.
    M[fs_u.node_count, fs_u.node_count] = 1

    return M, l


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)
    
    def on_boundary_vec(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return [1., 1.]
        else:
            return [0., 0.]

    f.interpolate(on_boundary_vec)
    
    return np.flatnonzero(f.values)

def solve_mastery(resolution, their_assembly=False, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given
    resolution. It should return both the solution
    :class:`~fe_utils.function_spaces.Function` and the :math:`L^2` error in
    the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)

    #Set up quadratic vector finite element
    fe_quadratic = LagrangeElement(mesh.cell, degree=2)
    fe_quadratic = VectorFiniteElement(fe_quadratic, dimension=2)
    fs_quadratic = FunctionSpace(mesh, fe_quadratic)

    #Set up linear finite element
    fe_linear = LagrangeElement(mesh.cell, degree=1)
    fs_linear = FunctionSpace(mesh, fe_linear)

    # Create a function to hold the analytic quadratic solution for comparison purposes.
    u_analytic = Function(fs_quadratic)
    u_analytic.interpolate(lambda x: [-2 * pi * sin(2*pi*x[1]) * (1 - cos(2*pi*x[0])), 2 * pi * sin(2*pi*x[0]) * (1 - cos(2*pi*x[1]))])

    # Create a function to hold the analytic linear solution for comparison purposes.
    p_analytic = Function(fs_linear)

    #p=0
    p_analytic.interpolate(lambda x: 0)
    #p non-zero
    #p_analytic.interpolate(lambda x: 2 * pi**2 * sin(2*pi*x[0]) * sin(2*pi*x[1]))

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return u_analytic, p_analytic, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f_u = Function(fs_quadratic)

    #p=0
    f_u.interpolate(lambda x: [4 * pi**3 * sin(2*pi*x[1]) * (2 * cos(2*pi*x[0]) - 1), 4 * pi**3 * sin(2*pi*x[0]) * (-2*cos(2*pi*x[1]) + 1)])
    #p non-zero
    #f_u.interpolate(lambda x: [4 * pi**3 * sin(2*pi*x[1]) * (3 * cos(2*pi*x[0]) - 1), 4 * pi**3 * sin(2*pi*x[0]) * (-1*cos(2*pi*x[1]) + 1)])

    # Assemble the finite element system.:
    A, l = assemble(fs_quadratic, fs_linear, f_u)

    # Create the function to hold the solution.
    u = Function(fs_quadratic)
    p = Function(fs_linear)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csc_matrix(A)
    lu= splinalg.splu(A)
    x = lu.solve(l)

    u.values = x[:len(u_analytic.values)]
    p.values = x[len(u_analytic.values):]

    #Find integral of p over domain and adjust p coeffs to integrate to 0
    integrated_p = p.integrate()
    p.values[:] -= integrated_p

    # Compute the L^2 error in the solution for testing purposes.
    error = np.sqrt(errornorm(u_analytic, u)**2 + errornorm(p_analytic, p)**2)

    if return_error:
        u.values -= u_analytic.values
        p.values -= p_analytic.values

    # Return the solution and the error in the solution.
    return (u, p) , error


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument(
        "--analytic", action="store_true",
        help="Plot the analytic solution instead of solving the finite"
        " element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument(
        "resolution", type=int, nargs=1,
        help="The number of cells in each direction on the mesh."
    )
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_mastery(resolution, analytic, plot_error)

    u[0].plot()
