import numpy as np
from .quadrature import gauss_quadrature
from .finite_elements import VectorFiniteElement

def errornorm(f1, f2):
    """Calculate the L^2 norm of the difference between f1 and f2."""

    fs1 = f1.function_space
    fs2 = f2.function_space

    fe1 = fs1.element
    fe2 = fs2.element
    mesh = fs1.mesh

    # Create a quadrature rule which is exact for (f1-f2)**2.
    Q = gauss_quadrature(fe1.cell, 2*max(fe1.degree, fe2.degree))

    # Evaluate the local basis functions at the quadrature points.
    phi = fe1.tabulate(Q.points)
    psi = fe2.tabulate(Q.points)

    phi_dims = len(phi.shape)
    norm = 0.
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))

        # Compute the actual cell quadrature.
        f1_evaluated = np.dot(f1.values[nodes1], phi.T) 
        f2_evaluated = np.dot(f2.values[nodes2], psi.T)
        diff = f1_evaluated - f2_evaluated
        
        if phi_dims == 2:
            #FiniteElement
            pointwise_squares = diff**2
        else:
            pointwise_squares = np.einsum('ij,ij->j', diff, diff)
        norm += np.dot(pointwise_squares, Q.weights) * detJ
        
    return norm**0.5
