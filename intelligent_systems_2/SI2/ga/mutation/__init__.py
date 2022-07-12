from abc import abstractmethod as _to_override, ABCMeta as _META
from .. import Genome as _G, get_rng as _d_rng
from numpy.random import Generator as _Rng


class Mutation(metaclass=_META):
	
	def __init__(self, prob: float) -> None:
		if not hasattr(Mutation, "_gen"):
			Mutation._gen = _d_rng()
   
		assert prob >= 0.0 and prob <= 1.0
		self.__prob = prob
	
	@property
	def prob(self): return self.__prob
  
	def rng(self) -> _Rng: return Mutation._gen
	
	@_to_override
	def _mutate(self, genome: _G) -> _G:...
 
	def mutate(self, genome: _G) -> _G: return self._mutate(genome)
 
 
from .adjacentswapmutation import *
from .randomnegationmutation import *
from .randomswapmutation import *
from .sliceinversionmutation import *