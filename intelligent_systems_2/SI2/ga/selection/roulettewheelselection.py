from typing import Callable as _C, List as _L
from .. import Genome as _G, Value as _V
from . import Selection as _Base
from functools import reduce as _reduce
import numpy as _np

class RouletteWheelSelection(_Base):
	def __init__(self, fitness_function: _C[[_G], _V]) -> None:
		super().__init__(fitness_function)
  
	def _select(self, population: _L[_G], count: int) -> _L[_G]:
		mapped = self.rank_population(population)
		_offset = abs(min(mapped.values())) + abs(max(mapped.values()))
		rank_sum: _V =_reduce(lambda v1, v2: v1 + v2, mapped.values()) + len(population) * _offset
		probs = _np.array(list(map(lambda fit_value: (fit_value + _offset)/(rank_sum), mapped.values())))
		probs /= probs.sum()
		chosen :_np.ndarray =self.rng().choice(_np.array(list(mapped.keys()), dtype=_G), size=count, p=probs, replace=False)
		return chosen.tolist()

