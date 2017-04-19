import scipy.sparse as sps
import sympy as sym
import numpy as np
from HopfConstructions_Functions import *
from HopfClass import *
from Frobenius_Tools import HigmanTrace
    
#Given two spaces returns their tensor product with as much structure as possible
def Tensor_Product(A,B):
    _type = Tensor_Type(A,B)
    name = A.name + '(T)' + B.name
    element_names=[]
    for iA in range(0,A.dim):
        for iB in range(0,B.dim):
            element_names.append(A.element_names[iA]+'(T)'+B.element_names[iB])
    if _type == 'HopfAlgebra':
        mult=Tensor_Mult(A,B)
        comult=Tensor_Comult(A,B)
        counit=np.kron(A.counit,B.counit)
        antipode = sps.kron(A.antipode,B.antipode)
        out = HopfAlgebra(name,element_names,mult,comult,counit,antipode)
        if A.int_flag != 'no' and B.int_flag!='no':
            out.Input_Integral(np.kron(A.int,B.int))
    elif _type == 'BiAlgebra':
        mult=Tensor_Mult(A,B)
        comult=Tensor_Comult(A,B)
        counit=np.kron(A.counit,B.counit)
        out = BiAlgebra(name,element_names,mult,comult,counit)
    elif _type == 'Algebra':
        mult=Tensor_Mult(A,B)
        out = Algebra(name,element_names,mult)
    elif _type == 'CoAlgebra':
        comult=Tensor_Comult(A,B)
        counit=np.kron(A.counit,B.counit)
        out = CoAlgebra(name,element_names,comult,counit)
    return out

#Take a Hopf algebra and element J and J^{-1} as inputs and returns the Drinfeld twist of the coalgebra structure J and J^{-1} are input as np.arrays in H\ot H
def Drinfeld_Twist(H,J,JI,twist_name):
    comult = Drinfeld_Twist_Comult(H,J,JI)
    name = H.name + '('+twist_name+')'
    antipode = Drinfeld_Twist_Antipode(H,J,JI)
    out = HopfAlgebra(name,H.element_names,H.mult,comult,H.counit,antipode)
    out.Input_Integral(H.int)
    return out
    
#Takes as input a Hopf algebra and outputs the adjoint module as a module algebra
def Left_Adjoint_Module(H):
    action = Left_Adjoint_Action(H)
    return ModuleAlgebra('ad'+H.name,H.name,action,H.element_names,H.mult)

#Given a Hopf algebra H and an H module algebra A constructs the smashed product A#H
def Left_Smash_Product(A,H):
    mult = Left_Smash_Product_Mult(A,H)
    ele_names = Left_Smash_Element_Names(A,H)
    return Algebra(A.name+'#'+H.name,ele_names,mult)

def Dual_Hopf_Algebra(H):
    mult = H.comult.transpose()
    comult = H.comult.transpose()
    antipode = H.antipode.transpose()
    element_names = Dual_Element_Names(H.element_names)
    counit = np.zeros((H.dim),dtype=complex)
    counit[0]=1
    return HopfAlgebra(H.name+'^*',element_names,mult,comult,counit,antipode)
  