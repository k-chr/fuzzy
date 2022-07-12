from abc import abstractmethod as _to_override, ABCMeta as _META
from .. import Genome as _G, get_rng as _d_rng
from numpy.random import Generator as _Rng
from typing import Tuple as _Tup


class Crossover(metaclass=_META):
	
	def __init__(self) -> None:
		if not hasattr(Crossover, "_gen"):
			Crossover._gen = _d_rng()

	def rng(self) -> _Rng: return Crossover._gen
   
	@_to_override
	def _crossover(self, first_parent: _G, second_parent: _G) -> _Tup[_G, _G]:...
 
	def crossover(self, first_parent: _G, second_parent: _G) -> _Tup[_G, _G]: return self._crossover(first_parent, second_parent)
 

from .kpointcrossover import KPointCrossover
from .shufflecrossover import ShuffleCrossover
from .randomrespectfulcrossover import RandomRespectfulCrossover