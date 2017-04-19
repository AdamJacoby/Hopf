import sympy as sp
import scipy.sparse as sps
from HopfClass import *
import numpy as np
import sympy as sym
from Frobenius_Tools import HigmanTrace
from HopfConstructions_Functions import CreateBasisVectors

#Given U the change of basis matrix twiost the structure matrixes to be wirth respect to the new basis
#U(original basis element)=new basis element
def ChangeBasis(A,U,UI):
    U = sps.csr_matrix(U.tolist())
    UI= sps.csr_matrix(UI.tolist())
    if 'Algebra' in A._type:
        mult = U.dot(A.mult.dot(sps.kron(UI,UI)))
    if 'Module' in A._type:
        ring_dim=self.ring.dim
        action = U.dot(A.action)
        ring_id = sps.Identity(ring_dim,format='csr')
        action = U.dot(action.dot(sps.kron(ring_id,UI)))
    if A._type=='HopfAlgebra' or A._type=='BiAlgebra' or A._type=='CoAlgebra':
        comult = sps.kron(U,U).dot(A.comult.dot(UI))
        counit = A.counit.dot(UI)
    if A._type == 'Algebra':
        out = Algebra(A.name,A.element_names,mult)
    if A._type == 'CoAlgebra':
        out = CoAlgebra(A.name,A.element_names,comult,counit)
    if A._type == 'BiAlgebra':
        out = BiAlgebra(A.name,A.element_names,mult,comult,counit)
    if A._type == 'HopfAlgebra':
        antipode = U.dot(A.antipode.dot(UI))
        out = HopfAlgebra(A.name,A.element_names,mult,comult,counit,antipode)
        if A.int_flag != 'no':
            A.Input_Integral(U.dot(A.Integral))
    if A._type == 'Module':
        out = Module(A.name,A.element_names,A.ring,action)
        if A._type == 'ModuleAlgebra':
            out = ModuleAlgebra(A.name,A.element_names,A.ring,action,mult)
    if 'Algebra' in A._type and A.casimir_flag !='no':
        out.Input_Casimir(sps.kron(U,U).dot(A.casimir))
    return out

#Computes a basis for the center of A as well as a complementary basis
def Center(A):
    dim = A.dim
    V_temp = CreateBasisVectors(dim)
    V=[]
    for vector in V_temp:
        V.append(vector.tolist())
    higman_trace=HigmanTrace(A)
    higman_trace = sym.Matrix(higman_trace.toarray())
    image = higman_trace.transpose().rref()
    image = image[0]
    rref_matrix = image.tolist()
    rref_matrix=filter(lambda a: a != [0]*dim, rref_matrix)
    center_dim = len(rref_matrix)
    change_of_basis = []
    past_index = 0
    complement = []
    for vector in rref_matrix:
        current_index = vector.index(1)
        for i in range(past_index+1,current_index):
            complement.append(i)
        change_of_basis.append(vector)
        past_index=current_index
    for i in range(current_index+1,dim):#add the remaining basis vectors if not finished in previous step
        complement.append(i)
    for index in complement:
        change_of_basis.append(V[index])
    U=np.transpose(np.array(change_of_basis,dtype=complex))
    UI=np.linalg.inv(U)
    return [U,UI,center_dim]