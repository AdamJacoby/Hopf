from HopfClass import *
###########################################################################################################
#Creates a new class of objects as elements of vector spaces/algebras/modules/bialgebras/Hopf algebras
#The primarry advantage to elements is that they can have names making them much easier to work with
##############################################################################################################

#Creates the most basic element that of a vector space
#It can be created either by entering the name associated to the algebra
#or by entering in a proper size vector
#names should be entered as sums of constant*elementname
#Elements have three properties name,vector and space
#All operations are inheartyed from the space
class VectorSpaceElement(object):
	#Note to save time this function does not name an unnamed vector
	def __init__(self,vector,space):
		if type(vector) !=str:
			self.vector = vector
		elif vector == '0':
			self.vector = np.zeros((space.dim))
		else:
			self.vector = np.zeros((space.dim),dtype=complex)
			temp = vector.split('+')
			for item in temp:
				element = item.split('*')
				for i in range(0,space.dim):
					if element[1] == space.element_names[i]:
						self.vector[i]=complex(element[0])
						break
		self.space = space
		self.name = 'Currently unnamed.'
	
	#Constructs the tensor product of two elements
	#Note the  tensor space must be defined for this to happen
	def Tensor(self,other):
		if self.space.type == 'HopfAlgebra' and other.space.type == 'HopfAlgebra':
			out_type = 'HopfAlgebra'
		elif self.space.type == 'Algebra' or other.space.type:
			out_type = 'Algebra'
		elif (self.space.type == 'ModuleAlgebra' or self.space.type == 'HopfAlgebra') and (other.space.type=='HopfALgebra' or other.space.type=='ModuleALgebra'):
			out_type = 'ModuleAlgebra'
		else:
			out_type = 'VectorSpace'
		out = np.zeros((self.space.dim*other.space.dim),dtype=complex)
		for i in range(0,self.space.dim):
			for j in range(0,other.space.dim):
				out[i*other.space.dim+j]=self.vector[i]*other.vector[j]
		return eval(out_type+'Element(out,AlgebraList[\''+self.space.name+'(T)'+other.space.name+'\'])')
	
	#Adds two elements
	def __add__(self,other):
		return eval(self.space.type+'Element(self.vector + other.vector,self.space)')
	
	#Subtracts two elements
	def __sub__(self,other):
		return eval(self.space.type+'Element(self.vector - other.vector,self.space)')
	
	#If the element does not have a name this function adds it
	def Name(self):
		if self.name == 'Currently unnamed.':
			self.name = ''
			for i in range(0,self.space.dim):
				if self.vector[i] != 0:
					self.name = self.name+'+'+str(self.vector[i])+'*'+self.space.element_names[i]
			if self.name == '':
				self.name = '0'
			return self.name
		else:
			return self.name

#Adds two new functions to the vector space element class
class AlgebraElement(VectorSpaceElement):
	#Defines the multiplication action multiplication comes from the space
	def __mul__(self,other):
		if type(other) == complex or type(other) == int or type(other) == float or type(other)==np.complex128:
			return eval(self.space.type+'Element(other*self.vector,self.space)')
		elif self.space==other.space:
			return eval(self.space.type+'Element(self.space.Mult(self.vector,other.vector),self.space)')
		elif other.space.type == 'Module':
			return ModuleAlgebraElement(other.space.Action(self.vector,other.vector),other.space)
		elif other.space.type == 'ModuleAlgebra':
			return ModuleAlgebraElement(other.space.Action(self.vector,other.vector),other.space)
	#Defines the expeniation action multiplication comes from the space
	def __pow__(self,other):
		out = self
		if other ==0:
			temp = np.zeros(self.vector.shape[0])
			temp[0]=1
			out = eval(self.space.type+'Element(temp,self.space)')
		else:
			for i in range(0,other-1):
				out = out*self
		return out

#Element class for a coalgebra
#Adds two functions onto a vector space element
#
class CoAlgebraElement(VectorSpaceElement):	
	#Defines the comultiplication all structure comes from the space
	def CoMult(self):
		if self.space.type == 'CoAlgebra':
			return CoAlgebraElement(self.space.CoMult(self.vector),AlgebraList[self.space.name+'(T)'+self.space.name])
		if self.space.type == 'HopfAlgebra':
			return HopfAlgebraElement(self.space.CoMult(self.vector),AlgebraList[self.space.name+'(T)'+self.space.name])
	#Defines the counit all structure comes from the space
	#output is an element of the base field currently the complex numbers
	def CoUnit(self):
		return self.space.CoUnit(self.vector)

#Has the functions of both an algebra element and a coalgebra element
#Only new function is the antipode
class HopfAlgebraElement(AlgebraElement,CoAlgebraElement):
	def Antipode(self):
		return HopfAlgebraElement(self.space.Antipode(self.vector),self.space)

#Element of a module currentl uncomplete !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class ModuleElement(VectorSpaceElement):
	def Act(self,ring_element):
		vector = self.space.Action(ring_element.vector,self.vector)
		return eval(self.space.type+'Element(vector,self.space)')
#Element of a module algebra has all functions as that of a Module element and an algebra element
#Also contains a function for the smash product of two elements
#Note the smash product must exist for this to happen
class ModuleAlgebraElement(ModuleElement,AlgebraElement):
	def Smash(self,other):
		vector = ProdVectorToTensorVector(self.vector,other.vector)
		return ModuleAlgebraElement(vector,AlgebraList[self.space.name+'#'+other.space.name])