from itertools import product
import scipy.sparse as sps
import sympy as sp
import numpy as np
from HopfConstructions_Functions import Left_Action_Matrix, Right_Action_Matrix, CreateBasisVectors, Tensor_Mult
from math import floor

#Conversts s scipymatrix to sympy then converts the charicteristic poly in sympy and outputs it
def CharPoly(M):
    temp = M.toarray()
    temp = sp.SparseMatrix(temp)
    return temp.berkowitz_charpoly()

#Given a polynomial poly with list of roots, roots, computes the multiplicity of the roots using derivatives where Der is a partial list of derivatives
def Multiplicity(poly,root,Der):
    D = Der #Start list of derivatives of poly
    i = 1
    flag = 'go'
    while flag == 'go':
        if i<=(len(D)-1):
            if D[i].eval(root)!=0:
                multiplicity = i
                flag = 'stop'
        else:
            temp = D[-1].diff()
            D.append(temp)
            if temp.eval(root) != 0:
                multiplicity = i
                flag = 'stop'
        i = i+1
    return [multiplicity,D]

def Devisors(n):#Computes all divisors of n less then sqrt(n)
    out = []
    for i in range(1,floor(n**.5)+1):
        if n%i==0:
            out.append(i)
    return out

#Takes a polynomial and factors outt the zeros at 0
def Remove_Zeros_At_Zero(poly):
    mon = poly.EM()
    mon = mon.as_expr()
    return sp.exquo(poly,mon)

#Computes the matrix corresponding to the higman trace: A\rightarrow A
def HigmanTrace(A):
    dim = A.dim
    if A.casimir_flag == 'no':
        A.GetCasimir()
    mult = A.mult
    V = CreateBasisVectors(dim)
    casimir = A.casimir
    higman_trace = sps.csr_matrix((dim,dim))
    for i,j in product(range(0,dim),range(0,dim)):#i,j corresponds to the baasis vector b^i\ot b^j
        if A.casimir[dim*i+j] !=0:
            higman_trace = higman_trace+Left_Action_Matrix(V[i],mult).dot(Right_Action_Matrix(V[j],mult))
    return higman_trace

#Given an algerbra A computes the mattrix corresponding to the left action of
#the image of the casimir squared under the multiplication map
def Compute_M(A):
    dim = A.dim
    if A.casimir_flag == 'no':
        A.GetCasimir()
    mult = A.mult
    casimir = A.casimir
    tensor_mult = Tensor_Mult(A,A)
    C = mult.dot(tensor_mult.dot(np.kron(A.casimir,A.casimir)))
    return Left_Action_Matrix(C,A.mult)