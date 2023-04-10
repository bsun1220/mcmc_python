import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

from scipy.optimize import minimize, LinearConstraint, linprog
import pandas as pd
import sympy

def equal_conversion(A, b):
    A_tilde = np.hstack((A, np.identity(A.shape[0])))
    return A_tilde, b

def find_z(A, b, x_dim):
    n = A.shape[0]
    d = A.shape[1]
    for i in range(x_dim, d):
        A_ub = A.T * -1
        b_ub = np.zeros(d)
        
        A_eq = np.vstack((A.T[0:x_dim], A.T[i], b.T))
        
        b_eq = np.zeros(x_dim)
        b_eq = np.append(b_eq, [1,0])
        
        c = np.ones(n)
        
        res = linprog(
            c = c,
            A_ub = A_ub,
            b_ub = b_ub,
            A_eq = A_eq,
            b_eq = b_eq
        )
        if res["success"]:
            return np.matmul(A.T, res.x)
        
    return False 

def facial_reduction(z):
    d = z.shape[0]
    z = np.round(z, 3)
    indices = (z <= 0).nonzero()[0]
    
    matrix = None
    
    for val in indices:
        arr = np.eye(1, d, val)
        matrix = arr if matrix is None else np.vstack((matrix, arr))
    
    return matrix.T

def entire_facial_reduction_step(A, b, x_dim):
    z = find_z(A, b, x_dim)

    if isinstance(z, bool):
        return (A, b)
    
    V = facial_reduction(z)
    AV = np.matmul(A, V)

    _, ind = sympy.Matrix(AV).T.rref()
    
    proj = None
    for i in ind:
        arr = np.eye(1, A.shape[0], i)
        proj = arr if proj is None else np.vstack((proj, arr))
        
    newA = np.matmul(proj, AV)
    newB = np.matmul(proj, b)

    return entire_facial_reduction_step(newA, newB, x_dim)

def old_make_full_rank(mat, i):
    if np.linalg.matrix_rank(mat) == mat.shape[1]:
        return mat
    
    basis = np.zeros(mat.shape[1])
    basis[i] = 1
    new_mat = np.vstack((mat, basis))
    
    '''
    Use QR Decomposition to M.T / add I
    
    '''
    
    depend = np.linalg.matrix_rank(mat) == np.linalg.matrix_rank(new_mat)
    pref_mat = mat if depend else new_mat
    return make_full_rank(pref_mat, i + 1)

def make_full_rank(mat):
    Q, R = np.linalg.qr(mat.T, mode = "complete")
    dim_diff = Q.shape[0] - R.shape[1]
    if dim_diff == 0:
        return mat
    
    new_mat = np.identity(Q.shape[0])[:,  Q.shape[0] - dim_diff:]
    
    R = np.hstack((R, new_mat))
    return np.matmul(Q, R).T

def reduce_sampling(M, b, delta_dim):
    M_inv = np.linalg.inv(M)
    row = M_inv.shape[0] - delta_dim
    gamma = M_inv.shape[0] - b.shape[0]
    C = np.matmul(M_inv[row:, :M_inv.shape[1] - gamma], b)
    D = -1 * M_inv[row:, M_inv.shape[1] - gamma:]
    return D, C

def reduce_problem(A, b):
    newA, newB = equal_conversion(A, b)
    A_tilde, b_tilde = entire_facial_reduction_step(newA, newB, A[1].shape[0])
    if A_tilde.shape == newA.shape and b.shape == b_tilde.shape:
        info = {"A":A, "b":b}
        return info
    M = make_full_rank(A_tilde)
    reduced_A, reduced_b = reduce_sampling(M, b_tilde, A_tilde.shape[1] - A[1].shape[0])
    M_inv = np.linalg.inv(M)
    
    info = {"A":reduced_A, "b":reduced_b, "pb":b_tilde, "M_inv":M_inv}
    return info