# packages
from HopfClass_Functions import *
from scipy import linalg
import scipy.sparse as sps
import numpy as np
import os
import pickle

##################################################################################################################
#Create the global variable storing algebras
###################################################################################################################

global AlgebraList
AlgebraList = {}
###############################################################################################################################
#Create a class for algebras
#The multiplication matrix should have the form of an n by n^2 matrix
#corisponding to the multiplication matrix with lexiographical ordering on the tensor product
#############################################################################################################################

#Creates a Class for vector spaces
#For the addition and scalar multiplication action see the class VectorSpaceElement
class VectorSpace(object):
    def __init__(self,name,element_names):
        #name of vector space
        self.name = name
        #List of the names of the elements in the vertor space
        self.element_names = element_names
        #A flag denoting the type of space
        self._type = 'VectorSpace'
        #The dimension of the vector space
        self.dim = len(element_names)
        #adds the object to the list of known objects
        global AlgebraList
        AlgebraList[self.name]=self
        
    #Saves the object to a location of your choice
    #if no location is given save to the current workign dirrectory
    def Save(self,*location):
        if len(location)==0:
            location = os.getcwd()
        else:
            location = location[0]
        temp_path = os.path.join(location, 'Save_'+self.name)
        temp_file = open(temp_path,'w+')
        pickle.dump(self,temp_file)
        temp_file.close()

#Creates a class for an assocative algebra
class Algebra(VectorSpace):
    def __init__(self,name,element_names,mult):
        self.name = name
        #The matrix that encodes the multiplication a dim by dim^2 matrix
        self.mult = mult
        self.dim = len(element_names)
        self.element_names = element_names
        self._type = 'Algebra'
        # a flag stating wether this algebra is Frobenius
        #I may replace this with a diffrent class all together?????????????????????????????????????????????????????
        self.casimir_flag = 'no'
        #A place to store the normalized regular character
        self.nrchar_flag = 'no'
        self.nrchar= None
        #A variable for the casimir element if known
        self.casimir = None
        global AlgebraList
        AlgebraList[self.name]=self
    #Uses the multiplication matrix to multiply two vectors
    def Mult(self,a,b):
        temp = np.kron(a,b)
        out = self.mult.dot(temp)
        return out
    
    #Computes the normalized regular character
    def GetNRChar(self):
        V = CreateBasisVectors(self.dim)
        self.nrchar = np.zeros((self.dim),dtype=complex)
        for i in range(0,self.dim):
            temp = 0
            for j in range(0,self.dim):
                temp = temp + self.Mult(V[i],V[j])[j]
            self.nrchar[i] = temp / self.dim
        self.nrchar_flag = 'yes'
        
    #Uses linear algebra to get casimir element if we are dealing with a hopf algebra !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def GetCasimir(self):
        V = CreateBasisVectors(self.dim)
        self.casimir = np.zeros((self.dim**2),dtype=complex)
        equations = np.zeros((self.dim,self.dim),dtype=complex)
        if self.nrchar_flag == 'no':
            self.GetNRChar()
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                equations[i,j] = self.nrchar.dot(self.Mult(V[i],V[j]))
        for i in range(0,self.dim):
            solutions = np.zeros((self.dim),dtype=complex)
            solutions[i]=1
            temp = linalg.solve(equations,solutions)
            self.casimir = self.casimir + np.kron(V[i],temp)
            
    def Input_Casimir(self,casimir):
        self.casimir=casimir
        self.casimir_flag = 'yes'
        
###########################################################################################################################
#Defines the class of coalgebras comult should be a Dim by Dim^2 matrix with top as inputs and side as outputs
#counit representes the counit map and shold be the vector of [\epsilon(T_1),\epsilon(T_2),...,\epsilon(T_{Dim})]
###########################################################################################################################
class CoAlgebra(VectorSpace):
    def __init__(self,name,element_names,comult,counit):
        self.name = name
        #The variable that stores the comulttiplication matrix
        self.comult = comult
        self.dim = len(element_names)
        #stores the counit
        self.counit = counit
        self.element_names = element_names
        self._type = 'CoAlgebra'
        global AlgebraList
        AlgebraList[self.name]=self
    
    #Takes an element in vector form as input and outputs it image under the comultiplication
    #As always the tensor is expressed in terms of the lexiographical basis
    def CoMult(self,a):
        return self.comult.dot(a)
    
    #Takes an element in vector form as the input and outputs the counit evaulated at it
    #Result is an element of the base field
    def CoUnit(self,a):
        return self.counit.dot(a)

        
        
#####################################################################################################
#Creates a Bialgebra class simple an Algebra and a coalgebra (NOTE: As of now compatibility conditions are NOT checked)
#####################################################################################################
class BiAlgebra(CoAlgebra,Algebra):
    def __init__(self,name,element_names,mult,comult,counit):
        self.name = name
        self.mult = mult
        self.dim = len(element_names)
        self.comult = comult
        self.counit = counit
        self.element_names = element_names
        self._type = 'BiAlgebra'
        self.casimir_flag = 'no'
        self.nrcharflag = 'no'
        self.casimir=None
        self.nrchar = None
        global AlgebraList
        AlgebraList[self.name]=self

        
        
#############################################################################################################
#A biablgebra plus an Antipode (Note the antipode axiomes are NOT checked)
#the Antipode will be input as a Dim by Dim matrix corresponding to the antipode under the standard basis
################################################################################################################
class HopfAlgebra(BiAlgebra):
    def __init__(self,name,element_names,mult,comult,counit,antipode):
        self.name = name
        self.element_names = element_names
        self.mult = mult
        self.dim = len(element_names)
        self.comult = comult
        self.counit = counit
        self.antipode = antipode
        self.int_flag = 'no'
        self.casimir_flag = 'no'
        self.nrchar_glag = 'no'
        self.integral = None
        self.casimir = None
        self.nrchar = None
        self._type = 'HopfAlgebra'
        global AlgebraList
        AlgebraList[self.name]=self
    
    #Take an element expressed as a vector as an input and outputs the antipode evaluated at it
    def Antipode(self,a):
        return self.antipode.dot(a)
    
    #This code should eventually be expanded so that it computes it for you need to be careful goal is for this to be computed at creation
    #This should be input as a vector not a Hopf element need left and right integrals
    def Input_Integral(self,integral):
        self.integral = integral
        self.int_flag = 'yes'
    
    #Given the integral uses it to compute the Casimir element
    #This should eventually be done at creation but need to get integral code running first
    def GetCasimir(self):
        if self.int_flag == 'yes':
            temp = self.CoMult(self.integral)
            temp = TensorVectorToProdVector(temp,self,self)
            out = np.zeros((self.dim**2),dtype=complex)
            for item in temp:
                out = out + np.kron(item[0],self.Antipode(item[1]))
            self.casimir = out
            self.casimir_flag = 'yes'
        else:
            Algebra.GetCasimir(self)
        
            
##################################################################################################################################
#Creates a module the action should be in the form of the matrix representing the action map from ring tensor module to module
#interms of the standard basis
##################################################################################################################################
class Module(VectorSpace):
    def __init__(self,name,element_names,ring,action):
        self.dim = len(element_names)
        #The ring that acts on it stored as a string
        self.ring = ring
        self.action = action
        self.elements_names
        self.name = name
        self._type = 'Module'
        global AlgebraList
        AlgebraList[self.name]=self
    
    #Given two vectors representing a ring element and module element respectively
    #outputs the vector corresponding to the ring element acting on the vector space element
    def Action(self,ring_vector,module_vector):
        return self.action.dot(np.kron(ring_vector,module_vector))
    
#Creates a class for a module algebra which is exactly just a module class and algebra class
class ModuleAlgebra(Algebra,Module):
    def __init__(self,name,element_names,ring,action,mult):
        self.name = name
        self.mult = mult
        self.dim = len(element_names)
        self.element_names = element_names
        self.mult=mult
        self.action = action
        self.casimir_flag = 'no'
        self.nrchar_flag = 'no'
        self._type = 'ModuleAlgebra'
        self.casimir = None
        self.nrchar = None
        global AlgebraList
        AlgebraList[self.name]=self