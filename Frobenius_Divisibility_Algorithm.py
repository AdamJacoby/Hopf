import sympy as sp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg
from HopfConstructions_Functions import Tensor_Mult, Left_Action_Matrix
from math import floor
from Frobenius_Tools import *
from Algebra_Tools import Center

#Computes the matrix for the image of the casimir squared under the multiplication map acting on the center
def Comupute_Central_M(A):
    M = np.array(Compute_M(A).toarray())
    temp = Center(A)
    U = temp[0]
    center_dim = temp[2]
    UI = temp[1]
    M=UI.dot(M.dot(U))
    M = np.array(M.tolist())
    M = M[0:center_dim,0:center_dim]
    print M.shape
    return M

def Check_FD(A):
    dim = A.dim
    if A.casimir_flag == 'no':
        A.GetCasimir()
    mult = A.mult
    casimir = A.casimir
    tensor_mult = Tensor_Mult(A,A)
    C = mult.dot(tensor_mult.dot(np.kron(A.casimir,A.casimir)))
    C = AlgebraElement(C,A)
    temp = C**2-((dim**2)/1)*C
    for d  in divisors[1:]:
        temp = temp*C**2-((dim**2)/d)*C
    zeros = np.zeros(dim)
    if temp.vector==zeros:
        print 'Yes FD'
    else:
        print 'No FD'
    

def Degree_Of_Irreps_Char(A):
    dim = A.dim
    number_of_irreps = []
    sizes_of_irreps = []
    M = Compute_M(A)#Compute the matrix corresponding to tthe left action by C
    poly = CharPoly(M)
    Der = [poly]
    d = dim
    bound = floor(dim**.5)
    i=1
    while i <= bound:
        root_to_test = dim**2/i**2
        if poly.eval(root_to_test) == 0:
            sizes_of_irreps.append(i)
            temp = Multiplicity(poly,root_to_test,Der)
            number_of_irreps.append(temp[0]/i**2)
            d = d - temp[0]
            Der = temp[1]
            bound = floor(d**.5)
        i = i+1
    return [sizes_of_irreps,number_of_irreps]

def Degree_Of_Irreps_Higchar(A):
    mult = A.mult
    dim = A.dim
    number_of_irreps = []
    sizes_of_irreps = []
    M = Compute_M(C)#Compute the matrix corresponding to tthe left action by C
    higman_trace = HigmanTrace(A)/dim
    M = M.dot(higman_trace)
    poly = CharPoly(M)
    poly = Remove_Zeros_At_Zero(poly)
    print poly
    Der = [poly]
    d = dim
    bound = floor(dim**.5)
    i=1
    while i <= bound:
        root_to_test = dim**2/i**2
        if poly.eval(root_to_test) == 0:
            sizes_of_irreps.append(i)
            temp = Multiplicity(poly,root_to_test,Der)
            number_of_irreps.append(temp[0])
            d = d - temp[0]
            Der = temp[1]
            bound = floor(d**.5)
        i = i+1
    return [sizes_of_irreps,number_of_irreps]

#Uses a straight determinant computation
def Degree_Of_Irreps_Det(A):
    dim = A.dim
    sizes_of_irreps = []
    M=Compute_M(A)#Compute the matrix corresponding to tthe left action by C
    M=np.array(M.toarray())
    d = dim
    float_dim = float(dim)
    bound = floor(dim**.5)
    root_dim = bound
    i=1
    while i <= bound:
        n = float(i)
        value_to_test = float_dim**2/n**2
        if i == root_dim:
            error = ((float_dim**2*(2*n-1)/(n**2*(n-1)**2))**float_dim)/2
        else:
            error = ((float_dim**2*(2*n+1)/(n**2*(n+1)**2))**float_dim)/2
        if np.linalg.det(M-value_to_test*np.identity(dim)) < error:
            sizes_of_irreps.append(i)
            d = d-i**2
            bound = floor(d**.5)
        i = i+1
    return sizes_of_irreps

#Uses scipy Eigen value solver
def Degree_Of_Irreps_Eig(A):
    dim = A.dim
    M = Compute_M(A)#Compute the matrix corresponding to tthe left action by C
    M=np.array(M.toarray())#COnvert to a nonsparse numpy array
    temp =np.round(np.sqrt(dim**2/np.real(np.linalg.eigvals(M)))).tolist()
    Eigen_Vals = sorted(list(set(temp)))
    Multiplicities = []
    for Val in Eigen_Vals:
        Multiplicities.append(temp.count(Val)/Val**2)
    print 'The diminsions of the irreps'+str(Eigen_Vals)
    print 'With Corresponding Multiplicities'+str(Multiplicities)
    
def Degree_Of_Irreps_Center(A):
    dim = A.dim
    M = Comupute_Central_M(A)
    temp =np.round(np.sqrt(dim**2/np.real(np.linalg.eigvals(M)))).tolist()
    Eigen_Vals = sorted(list(set(temp)))
    Multiplicities = []
    for Val in Eigen_Vals:
        Multiplicities.append(temp.count(Val))
    print 'The diminsions of the irreps'+str(Eigen_Vals)
    print 'With Corresponding Multiplicities'+str(Multiplicities)