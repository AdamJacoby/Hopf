from HopfClass import *
import numpy as np
from itertools import product
import scipy.sparse as sps
#Creates the name vector for the Taft algebra
#g is the name of the group like
#x is the name of the g,1 primitive element
#basis is given by g^0x^0,g^2,...,g^1x^1,g^2x^1,...,g^{n-2}x^{n-1},g^{n-1}x^{n-1}
def Taft_Element_Names(n,g,x):
	out=[]
	for i in range(0,n):
		for j in range(0,n):
			out.append(g+'^'+str(j)+x+'^'+str(i))
	return out

#Created the multiplication matrix for the taft algebra
#Uses the same ordered basis that was used for element names
def Taft_Mult(n):
	poly = [0]*(n+1)
	poly[0]=1
	poly[n]=-1
	#Omega is the lsit of roots of unity where omega[0] is the primitive one underconsideration in the definition
	omega=np.roots(poly)
	out = np.zeros((n**2,n**4),dtype=complex)
	N=range(0,n)
	for i, j, k in product(N,N,N):# the iteration i,j,l,k corresponds to the input g^ix^j\ot g^kx^l
		for l in range(0,n-j):#only loops up to n-j since if j+l>=n the product should be 0
			out[((i+k)%n)+n*(j+l),i+n*j+(k+n*l)*n**2]=omega[(n-1)-(-j*k)%n]
	out = sps.csr_matrix(out.tolist(),dtype=complex)
	return out

#Constructs the comultiplication matrix inductively perhaps latter will want to do it in a closed form!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Taft_Comult(n,mult):
	dim = n**2
	out=np.zeros((dim**2,dim),dtype=complex)
	#compute list of powers of (x+g) between 0 and n-1
	xplusg=np.zeros(dim)
	xplusg[1]=1
	xplusg[n]=1
	temp = np.zeros(dim)
	temp[0]=1
	prods =[temp]
	N=range(1,n)
	for i in N:
		temp2 = np.kron(temp,xplusg)
		temp = mult.dot(temp2)
		prods.append(temp)
	N=range(0,n)
	for i, j in product(N,N):#i,j corresponds to g^ix^j
		for l in range(0,j+1):
			out[((i+l)%n)+(j-l)*n+(i+n*l)*dim,i+j*n]=prods[j][l+(j-l)*n]
	return out

def Taft_Counit(n):
	out = np.zeros((n),dtype=complex)
	for i in range(0,n):
		out[i]=1
	return out

def Taft_Antipode(n):
	dim=n**2
	poly = [0]*(n+1)
	N=range(0,n)
	poly[0]=1
	out=np.zeros((dim,dim),dtype=complex)
	poly[n]=-1
	omega=np.roots(poly)
	for i, j in product(N,N):#(i,j) corresponds to g^ix^j
		out[((-i)%n)+j*n]=(-1)**j*omega[(i*j)%n]#gives coefficent of g^{n-i}x^j
	out = sps.csr_matrix(out.tolist(),dtype=complex)
	return out
	
###################################################################################################
#Code For testing only	
######################################################################################################