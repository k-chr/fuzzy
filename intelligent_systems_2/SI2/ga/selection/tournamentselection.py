from typing import Callable as _C, List as _L
from .. import Genome as _G, Value as _V, PriorityQueue as _Heap
from . import Selection as _Base

from itertools import cycle as _cycle

class TournamentSelection(_Base):
	
	def __init__(self, fitness_function: _C[[_G], _V]) -> None:
		super().__init__(fitness_function)
	
	def _select(self, population: _L[_G], count: int) -> _L[_G]:
		fit_values = self.rank_population(population)  
		tournaments = [_Heap[_G]() for _ in range(count)]
		iter_tournaments = _cycle(tournaments)
  
		while fit_values:
			genome: _G = self.rng().choice(list(fit_values.keys()))
			value = fit_values.pop(genome, None)
			if value is not None:
				tournament = next(iter_tournaments)
				tournament.push(genome, value)			
	
		return [tournament.pop() for tournament in tournaments]