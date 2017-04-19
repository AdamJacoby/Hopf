import numpy as np
from scipy.sparse import csr_matrix
from itertools import product
from HopfClass import Algebra

def MartixAlgebra_ElementNames(n):
    ele_names=[]
    for i in range(0,n):
        for j in range(0,n):
            ele_names.append('T_'+str(i)+','+str(j))
    return ele_names

def MatrixAlgebra_Mult_Matrix(n):
    dim=n**2
    mult = np.zeros((dim,dim**2))
    for i,j,k in product(range(0,n),range(0,n),range(0,n)): #Corresponds to the product T_i,jT_j,k
        mult[i*n+k,dim*(i*n+j)+n*j+k]=1
    return csr_matrix(mult.tolist())

def MatrixAlgebra_Casimir(n):
    dim = n**2
    casimir = np.zeros((dim**2))
    for i,j in product(range(0,n),range(0,n)):#Coresponds to T_i,j\ot T_j,i
        casimir[dim*(n*i+j)+n*j+i]=n
    return casimir

def MatrixAlgebra(n):
    M = Algebra('M_'+str(n),MartixAlgebra_ElementNames(n),MatrixAlgebra_Mult_Matrix(n))
    M.Input_Casimir(MatrixAlgebra_Casimir(n))
    return M