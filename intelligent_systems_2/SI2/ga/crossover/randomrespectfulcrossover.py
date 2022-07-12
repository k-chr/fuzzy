from . import Crossover as _Base
from .. import Genome as _G
from typing import Tuple as _Tup
import numpy as _np

class RandomRespectfulCrossover(_Base):
	def __init__(self) -> None:
		super().__init__()
  
	def _crossover(self, first_parent: _G, second_parent: _G) -> _Tup[_G, _G]:
		null = 2
		child1 = first_parent.clone()
		child2 = second_parent.clone()
		for pos, locuses in enumerate(child1.get_locuses()):
			similarity_vector = _np.zeros_like(locuses)
			similarity_vector[child1[pos][:] != child2[pos][:]] = null
			similarity_vector[(child1[pos][:] == 1) & (child2[pos][:] == 1)] = 1
			
			arr1 = _np.where(similarity_vector == null, int(self.rng().random() > 0.5), similarity_vector)
			arr2 = _np.where(similarity_vector == null, int(self.rng().random() > 0.5), similarity_vector)
			child1[pos][:] = arr1
			child2[pos][:] = arr2
		
		return child1, child2