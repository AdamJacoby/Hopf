from sympy.combinatorics.named_groups import SymmetricGroup
from ExampleGroup_Functions import PermutationGroupInfo, MakeGroupAlgebra

def Symmetric_Group(n):
    temp = PermutationGroupInfo(SymmetricGroup(n))
    S = MakeGroupAlgebra('S_'+str(n),temp['element_names'],temp['mult'],temp['antipode'])
    return S
