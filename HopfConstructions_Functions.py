from itertools import product
import scipy.sparse as sps
import numpy as np

#Constructs the twist isomorphism from A\ot B to B\otA
#Columns correspond to input rows to outputs
def Twist_old(dimA,dimB):
    dim=dimA*dimB
    out = sps.csr_matrix((dim,dim),dtype=np.int8)
    for i,j in product(range(0,dimA),range(0,dimB)):#i is the index of A and j is the index of B
        out[i+j*dimA,dimB*i+j]=1
    return out

def Twist(dimA,dimB):
    dim=dimA*dimB
    zeros = np.zeros(dim,dtype=np.int8)
    out = zeros
    out[0,0]=1
    out = sps.csr_matrix(out)
    for i in range(1,dimB):
        temp_vector = zeros
        temp_vector[dimB*i]=1
        temp_vector = sps.csr_matrix(temp_vector)
        out = sps.vstack([out,temp_vector])
    for j in range(1,dimA):
            for i in range(0,dimB)):#i is the index of A and j is the index of B
        temp_vector = zeros
        temp_vector[dimB*i+j]=1
        temp_vector = sps.csr_matrix(temp_vector)
        out = sps.vstack([out,temp_vector])
    return out
    
#Take the kron product of more then two sparse matrixes at once
def Spare_Multi_Kron(*arg):
    out = arg[0]
    list = arg[1:len(arg)]
    for temp in list:
        out =sps.kron(out,temp)
    return out
    

#creates a list of the standard basis vectors for a vector space of dimension dim
def CreateBasisVectors(dim):
    out = []
    for i in range(0,dim):
        temp = np.zeros((dim),dtype=complex)
        temp[i]=1
        out.append(temp)
    return out

#creates a list of the standard basis vectors for a vector space of dimension dim
def CreateSparseBasisVectors(dim):
    out = []
    for i in range(0,dim):
        temp = [0]*dim
        temp[i]=1
        out.append(sps.csr_matrix(temp,dtype=np.int8))
    return out


#given an imput of (dimA1,dimA2,...,dimAn,d) for d an interger it computes the map id_1\ot id_2\ot id_{d-1}\ot \tau\ot id_{d+2}\ot..\id_n
def Center_Twist(*arg):
    length = len(arg)
    d = arg[length-1]
    spaces = arg[:length-1]
    Ldim=1#Will be sum_{i=1}^{d-1} dim A_i
    for dim in spaces[:d-1]:
        Ldim = Ldim*dim
    Rdim=1#Will be sum_{i=1}^{d-1} dim A_i
    for dim in spaces[d+1:]:
        Rdim = Rdim*dim
    return Spare_Multi_Kron(sps.identity(Ldim,dtype=complex,format='csr'),Twist(spaces[d-1],spaces[d]),sps.identity(Rdim,dtype=complex,format='csr'))
    
#Takes two algebras as inputs and outputs the matrix for their tensor product
def Tensor_Mult(A,B):
    return sps.kron(A.mult,B.mult).dot(Center_Twist(A.dim,B.dim,A.dim,B.dim,2))

#Takes two coalgebras and returns the comultiplication matrix of their tensor product
def Tensor_Comult(A,B):
    return Center_Twist(A.dim,A.dim,B.dim,B.dim,2).dot(sps.kron(A.comult,B.comult))
    
#Given an element a in a space A with multiplication matrix mult return the matrix for the map b mapsto ab (perhaps construct it using hstack command latter)
def Left_Action_Matrix(element,mult):
    dim = len(element)
    element = sps.csr_matrix(element)
    V = CreateSparseBasisVectors(dim)
    columns=[]
    for k in range(0,dim):
        columns.append(sps.kron(element,V[k]).transpose())
    temp = sps.hstack(columns)
    return mult.dot(temp)

#Given an element a in a space A with multiplication matrix mult return the matrix for the map b mapsto ab (perhaps construct it using hstack command latter)
def Right_Action_Matrix(element,mult):
    dim = len(element)
    element = sps.csr_matrix(element)
    V = CreateSparseBasisVectors(dim)
    columns=[]
    for k in range(0,dim):
        columns.append(sps.kron(V[k],element).transpose())
    temp = sps.hstack(columns)
    return mult.dot(temp)


#Determins the strongest structure that can be put on A\ot B
def Tensor_Type(A,B):
    _type='VectorSpace'
    if A._type == 'HopfAlgebra':
        if B._type=='HopfAlgebra' or B._type=='Algebra' or B._type=='Bialgebra' or (B._type=='ModuleAlgebra' and B.ring==A.name) or (B._type=='Module' and B.ring==A.name) or B._type=='CoAlgebra':
            type=B._type
    elif A._type == 'BiAlgebra':
        if B._type=='HopfAlgebra':
            _type = 'BiAlgebra'
        elif B._type=='Algebra' or B._type=='Bialgebra' or (B._type=='ModuleAlgebra' and B.ring==A.name) or (B._type=='Module' and B.ring==A.name) or B._type=='CoAlgebra':
            _type == B._type
    elif A._type == 'ModuleAlgebra':
        if (B._type=='ModuleAlgebra' and B.ring==A.ring) or ((B._type=='HopfAlgebra' or B._type=='BiAlgebra') and B.name==A.ring):
            _type='ModuleAlgebra'
        elif B._type=='Algebra' or (B._type=='Module' and B.ring==A.ring):
            _type==B._type
    elif A._type=='Module':
        if (B._type=='Module' or B._type =='ModuleALgebra') and A.ring==B.ring:
            _type=='Module'
        elif (B._type =='HopfAlgebra' or B._type=='BiAlgebra') and A.ring==B.name:
            _type=='Module'
    elif A._type=='CoAlgebra':
        if B._type=='HopfAlgebra' or B._type=='BiAlgebra' or B._type=='CoAlgebra':
            _type=='CoAlgebra'
    return _type

#Constructs the comultiplication matrix of H twisted by J
def Drinfeld_Twist_Comult(H,J,JI):
    tensormult = Tensor_Mult(H,H)
    L_J=Left_Action_Matrix(J,tensormult)
    R_JI=Right_Action_Matrix(JI,tensormult)
    return (L_J.dot(R_JI)).dot(H.comult)
            
#Constructs the antipode matrix of H twisted by J
def Drinfeld_Twist_Antipode(H,J,JI):
    Id=sps.identity(H.dim,dtype=complex,format='csr')
    UJ=H.mult.dot(sps.kron(Id,H.antipode).dot(J))
    UJI=H.mult.dot(sps.kron(H.antipode,Id).dot(JI))
    L_UJ=Left_Action_Matrix(UJ,H.mult)
    R_UJI=Right_Action_Matrix(UJI,H.mult)
    return R_UJI.dot(L_UJ.dot(H.antipode))
    
#Takes a Hopf algebra and returns the action matrix for the adjoint action 
def Left_Adjoint_Action(H):
    dim = H.dim
    Id = sps.identity(dim,dtype=complex,format='csr')
    twist = Center_Twist(dim,dim,dim,2)
    return H.mult.dot(sps.kron(H.mult,Id).dot(twist.dot(sps.kron(H.comult,Id))))

#Given A a H-module algebra and H a Hoipf algebra returns the multiplication matrix of A#H
def Left_Smash_Product_Mult(A,H):
    IdA=sps.identity(A.dim,dtype=complex,format='csr')
    IdH=sps.identity(A.dim,dtype=complex,format='csr')
    IdAH=sps.identity(A.dim*H.dim,dtype=complex,format='csr')
    left_comult=sps.kron(sps.kron(IdA,H.comult),IdAH)
    twist = Center_Twist(A.dim,H.dim,H.dim,A.dim,H.dim,3)
    act_and_hmult=Spare_Multi_Kron(IdA,A.action,H.mult)
    amult = sps.kron(A.mult,IdH)
    return amult.dot(act_and_hmult.dot(twist.dot(left_comult)))

#Returns the element_names list of A$H
def Left_Smash_Element_Names(A,H):
    out = []
    for iB in range(0,H.dim):
        for iA in range(0,A.dim):
            out.append(A.element_names[iA]+'#'+H.element_names[iB])
    return out

#Constructs the element names of the dual
def Dual_Element_Names(ele_names):
    out = []
    for ele in ele_names:
        out.append('P_('+ele+')')
    return out