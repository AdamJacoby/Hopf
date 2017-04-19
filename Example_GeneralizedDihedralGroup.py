#Constructs the group algrebra of the group G={g,x|g^q=x^p=1,xg=g^rx}
#p and q should be prime where p should divide q-1 and r should be an element of \ZZ_q of order p

from ExampleGroup_Functions import *
from HopfClass import HopfAlgebra
import numpy as np
import scipy.sparse as sps
from itertools import product

#Cronstrust the dihedrqal group {g^n=1,x^2=1,gx=xg^{-1}}
# g^ix^j corresponds to the basis vector with a 1 in position n*i+j
def GeneralizedDihedralGroup_Mult_Matrix(p,q,r):
	dim = p*q
	mult = np.zeros((dim,dim**2),dtype=complex)
	for i,j in product(range(0,q),range(0,p)):#Corresponds to the firstt element in the product g^ix^j
		for k,l in product(range(0,q),range(0,p)):#Corresponds to the firstt element in the product g^kx^l
			mult[p*((i+r*k)%q)+(k+l)%p,dim*(i*p+j)+p*k+l]=1 #Element corresponding to g^{i+r*k}x^{k+l} as the output
	mult = sps.csr_matrix(mult.tolist(),dtype=complex)
	return mult

#Constructts the antipode matrix for the dihedral group
def GeneralizedDihedralGroup_Antipode(p,q,r):
	dim = p*q
	antipode = np.zeros((dim,dim),dtype=complex)
	for i,j in product(range(0,q),range(0,p)):#Corresponds to input of g^ix^j
		inv=(((r**j)%q)**(q-2))%q#Compute r^j inverse mod q
		antipode[((-i*inv)%q)*p+((-j)%p),i*p+j]=1#Corresponds to output g^{(-1)^{j+1}i}x^(-j%p)
	antipode = sps.csr_matrix(antipode.tolist(),dtype=complex)
	return antipode

#Contructs the element name list were ele_name_g takes the role of g in discriptions above
#                                and  ele_name_x takes the role of x in discriptions above
def GeneralizedDihedralGroup_Element_Names(p,q,ele_name_g,ele_name_x):
	out = []
	for i in range(0,q):#corresponds to i in g^ix^j
		for j in range(0,p):#corresponds to j in g^ix^j
			out.append(ele_name_g+'^'+str(i)+ele_name_x+'^'+str(j))#Gives the name
	return out

#Constructs the dihedral group {g^n=1,x^2=1,gx=xg^{-1}} where g is names ele_name_g and x is named ele_name_x
def GeneralizedDihedralGroup(p,q,r,element_name_g,element_name_x):
	dim=p*q
	mult = GeneralizedDihedralGroup_Mult_Matrix(p,q,r)
	comult = Group_Comult_Matrix(dim)
	counit = Group_Counit(dim)
	int = Group_Integral(dim)
	antipode = GeneralizedDihedralGroup_Antipode(p,q,r)
	name = 'B_'+str(p)+','+str(q)+','+str(r)
	element_names = GeneralizedDihedralGroup_Element_Names(p,q,element_name_g,element_name_x)
	out = HopfAlgebra(name,element_names,mult,comult,counit,antipode)
	out.Input_Integral(int)#Sets the integral
	return out