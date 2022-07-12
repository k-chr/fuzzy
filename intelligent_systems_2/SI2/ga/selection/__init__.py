from typing import Callable as _C, Dict as _D, List as _L
from .. import Value as _V, Genome as _G, get_rng as _d_rng
from abc import abstractmethod as _to_override, ABCMeta as _META
from numpy.random import  Generator as _Rng


class Selection(metaclass=_META):
	def __init__(self, fitness_function: _C[[_G], _V]) -> None:
		self.__callback = fitness_function
		if not hasattr(Selection, "_gen"):
			Selection._gen = _d_rng()
	
	def rng(self) -> _Rng: return Selection._gen
	
	@_to_override
	def _select(self, population: _L[_G], count: int)->_L[_G]:...
 
	def select(self, population: _L[_G], count: int)->_L[_G]: 
		return self._select(population, count)
	
	def rank_population(self, population: _L[_G])-> _D[_G, _V]: 
		return dict(map(lambda genome: (genome, self.__callback(genome)), population))
 
from .rankselection import RankSelection
from .roulettewheelselection import RouletteWheelSelection
from .tournamentselection import TournamentSelection 