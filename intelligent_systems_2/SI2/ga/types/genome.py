from typing import List as _L, Tuple as _Tup
from .chromosome import Chromosome as _Ch, ChromosomePrecision as _ChP
from .. import Value as _V
import numpy as _np

class Genome:
    
	def __init__(self, rng: _np.random.Generator, precision: _ChP =_ChP.SINGLE, constraints: _L[_Tup[_V, _V]]= None, chromosomes:_L[_Ch]=None, num_of_params: int =None) -> None:
		self.__chromosomes: _L[_Ch] =[]
		self.__rng: _np.random.Generator = rng
		self.__precision = precision
		if chromosomes: self.__chromosomes = chromosomes
		elif (constraints) and len(constraints) > 1:
			self.__chromosomes = [_Ch(value_range, self.__rng.random(), self.__precision) for value_range in constraints]
		elif constraints and len(constraints) == 1 and num_of_params:
			constraints *= num_of_params
			self.__chromosomes = [_Ch(value_range, self.__rng.random(), self.__precision) for value_range in constraints]
		elif num_of_params:
			constraints = [(0.0, 1.0)] * num_of_params
			self.__chromosomes = [_Ch(value_range, self.__rng.random(), self.__precision) for value_range in constraints]
		else:
			self.__chromosomes = [_Ch((0.0, 1.0)), self.__rng.random(), self.__precision]
   
	def clone(self):
		dumped = [chromosome.clone() for chromosome in self.__chromosomes]
		obj = Genome(self.__rng, self.__precision, chromosomes=dumped)
		return obj

	def get_locuses(self):
		return [chromosome.locuses for chromosome in self.__chromosomes]

	def __getitem__(self, index):
		return self.__chromosomes[index]

	def __repr__(self) -> str:
		return "|".join(map(lambda chromosome: f"{chromosome}", self.__chromosomes))

	def decode_genetic_information(self):
		return [chromosome.value for chromosome in self.__chromosomes]
