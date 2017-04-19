import numpy as np
from scipy.sparse import csr_matrix
from itertools import product
from HopfClass import HopfAlgebra
import sympy.combinatorics as comb

#Constructs the integral of a group algebra
def Group_Integral(dim):
	out = np.zeros((dim),dtype=complex)
	for i in range(0,dim):
		out[i]=1
	return out

#Constructs the counit of a group algebra
def Group_Counit(dim):
	out = np.zeros((dim),dtype=complex)
	for i in range(0,dim):
		out[i]=1
	return out

#Constructs the comultiplication of a group algebra
def Group_Comult_Matrix(dim):
	temp=np.zeros((dim**2,dim),dtype=complex)
	for i in range(0,dim):
		temp[i+dim*i,i]=1
	return csr_matrix(temp.tolist(),dtype=complex)

#Takes the name,element names multiplication matrix and antipode matrix and outputs the corresponding group algebr with the identity
def MakeGroupAlgebra(name,ele_names,mult,antipode):
    dim = len(ele_names)
    G = HopfAlgebra(name,ele_names,mult,Group_Comult_Matrix(dim),Group_Counit(dim),antipode)
    G.Input_Integral(Group_Integral(dim))
    return G


#Given a sympy permutation group object out puts the ocrresponding multiplication matrix
def PermutationGroupInfo(P):
    Elements = list(P.elements)
    dim = len(Elements)
    Permutation_Size=max(list(Elements[0]))
    idenity = comb.Permutation(Permutation_Size)
    Elements.remove(idenity)
    Elements.insert(0,idenity)
    ele_names = []
    for Element in Elements:
        ele_names.append(str(list(Element)).replace(" ",""))
    mult = np.zeros((dim,dim**2))
    for i, j in product(range(0,dim),range(0,dim)):#Corresponds to g_i*g_j
        k = Elements.index(Elements[i]*Elements[j])#finds the lockation of g_i*g_j in the list Elements
        mult[k,i*dim+j]=1
    mult = csr_matrix(mult.tolist(),dtype=np.int8)
    antipode = np.zeros((dim,dim))
    for i in range(0,dim):
        inverse = Elements[i]**-1
        j = Elements.index(inverse)
        antipode[j,i]=1
    antipode = csr_matrix(antipode.tolist(),dtype=np.int8)
    return {'element_names':ele_names,'mult':mult,'antipode':antipode}