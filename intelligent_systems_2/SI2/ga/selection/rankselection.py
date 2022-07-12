from typing import Callable as _C, List as _L
from .. import Genome as _G, Value as _V, PriorityQueue as _Heap
from . import Selection as _Base

class RankSelection(_Base):
    
	def __init__(self, fitness_function: _C[[_G], _V]) -> None:
		super().__init__(fitness_function)
  
	def _select(self, population: _L[_G], count: int) -> _L[_G]:
		fit_values = self.rank_population(population)
		q = _Heap[_G]()
		for genom, value in fit_values.items():
			q.push(genom, value) 
		return [q.pop() for _ in range(count)]

