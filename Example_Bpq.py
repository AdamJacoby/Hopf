#This code constructs the following Hopf algebra
#as an algebra it is KG for G = {sigma,tau,a,b|simga^p=tau^p=a^q=b^q=1[sigma,b]=[a,tau]=[ba]=[sigma,tau]=1,sigma*a=a^r*sigma,tau*b=b^r*tau}

import numpy as np
import scipy.sparse as sps
from HopfClass import HopfAlgebra
from HopfConstructions import Tensor_Product, Drinfeld_Twist
from itertools import product
from Example_GeneralizedDihedralGroup import GeneralizedDihedralGroup



#Construnt a list of roots of unity the first one 1 the nextt is ptimitive
def Roots_Of_Unity(n):
	out = []
	for i in range(0,n):
		out.append(np.exp(i*2j*np.pi/n))
	return out

#two numbers p and q such that p|q-1 and roots a list of pth roots of unity out puts the twist J and J^{-1} as [J,J^{-1}]
def Bpq_Twist(p,q):
	dim = (p*q)**2
	omega=Roots_Of_Unity(p)
	J=np.zeros((dim**2),dtype=complex)
	JI=np.zeros((dim**2),dtype=complex)
	for i,j in product(range(0,p),range(0,p)):#Corresponds to the element tau^i\ot sigma^j
		J[dim*i+j]=(1/p)*omega[(-i*j)%p]
		JI[dim*i+j]=(1/p)*omega[(i*j)%p]
	return [J,JI]

def Bpq(p,q,r):
	G1=GeneralizedDihedralGroup(p,q,r,'a','sigma')
	G2=GeneralizedDihedralGroup(p,q,r,'b','tau')
	G=Tensor_Product(G1,G2)
	temp = Bpq_Twist(p,q)
	J=temp[0]
	JI=temp[1]
	return Drinfeld_Twist(G,J,JI,'J')