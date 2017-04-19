from ExampleGroup_Functions import *
from HopfClass import HopfAlgebra
import numpy as np
import scipy.sparse as sps
from itertools import product

#Constructs the multiplication matrix of the cyclic group of order dim in csr sparse format
def CyclicGroup_Mult_Matrix(dim):
	mult = np.zeros((dim,dim**2),dtype=complex)
	N=range(0,dim)
	for i,j in product(N,N):
		mult[(i+j)%dim,i*dim+j]=1
	mult = sps.csr_matrix(mult.tolist(),dtype=complex)
	return mult

def CyclicGroup_Antipode(dim):
	antipode = np.zeros((dim,dim),dtype=complex)
	for i in range(0,dim):
		antipode[i,(-i)%dim]=1
	antipode = sps.csr_matrix(antipode.tolist(),dtype=complex)
	return antipode

def CyclicGroup_Element_Names(dim,ele_name):
	out = []
	for i in range(0,dim):
		out.append(ele_name+'^'+str(i))
	return out
	
def CyclicGroup(dim,element_name):
	mult = CyclicGroup_Mult_Matrix(dim)
	comult = Group_Comult_Matrix(dim)
	counit = Group_Counit(dim)
	int = Group_Integral(dim)
	antipode = CyclicGroup_Antipode(dim)
	name = 'C_'+str(dim)
	element_names = CyclicGroup_Element_Names(dim,element_name)
	out = HopfAlgebra(name,element_names,mult,comult,counit,antipode)
	out.Input_Integral(int)
	return out