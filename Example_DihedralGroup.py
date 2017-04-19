from ExampleGroup_Functions import *
from HopfClass import HopfAlgebra
import numpy as np
import scipy.sparse as sps
from itertools import product

#Cronstrust the dihedrqal group {g^n=1,x^2=1,gx=xg^{-1}}
# g^ix^j corresponds to the basis vector with a 1 in position i*2+j
def DihedralGroup_Mult_Matrix(n):
	dim = 2*n
	mult = np.zeros((dim,dim**2),dtype=complex)
	N=range(0,n)
	for i,j in product(N,[0,1]):#Corresponds to the element in the product g^ix^j
		for k,l in product(N,[0,1]):#Corresponds to the element in the product g^kx^l
			mult[2*((i+((-1)**j)*k)%n)+(j+l)%2,dim*(i*2+j)+2*k+l]=1 #Element corresponding to g^{i+(-1)^j*k}x^{k+l} as the output
	mult = sps.csr_matrix(mult.tolist(),dtype=complex)
	return mult

#Constructts the antipode matrix for the dihedral group
def DihedralGroup_Antipode(n):
	dim = 2*n
	antipode = np.zeros((dim,dim),dtype=complex)
	for i,j in product(range(0,n),[0,1]):#Corresponds to input of g^ix^j
		antipode[2*(((-1)**(j+1)*i)%n)+j,i*2+j]=1#Corresponds to output g^{(-1)^{j+1}i}x^j
	antipode = sps.csr_matrix(antipode.tolist(),dtype=complex)
	return antipode

#Contructs the element name list were ele_name_g takes the role of g in discriptions above
#                                and  ele_name_x takes the role of x in discriptions above
def DihedralGroup_Element_Names(n,ele_name_g,ele_name_x):
	out = []
	for i in range(0,n):#corresponds to i in g^ix^j
		for j in [0,1]:#corresponds to j in g^ix^j
			out.append(ele_name_g+'^'+str(i)+ele_name_x+'^'+str(j))#Gives the name
	return out

#Constructs the dihedral group {g^n=1,x^2=1,gx=xg^{-1}} where g is names ele_name_g and x is named ele_name_x
def DihedralGroup(n,element_name_g,element_name_x):
	dim=2*n
	mult = DihedralGroup_Mult_Matrix(n)
	comult = Group_Comult_Matrix(dim)
	counit = Group_Counit(dim)
	int = Group_Integral(dim)
	antipode = DihedralGroup_Antipode(n)
	name = 'D_'+str(n)
	element_names = DihedralGroup_Element_Names(n,element_name_g,element_name_x)
	out = HopfAlgebra(name,element_names,mult,comult,counit,antipode)
	out.Input_Integral(int)#Sets the integral
	return out