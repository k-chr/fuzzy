from . import Crossover as _Base
from .. import Genome as _G
from typing import Tuple as _Tup


class ShuffleCrossover(_Base):
    
	def __init__(self) -> None:
		super().__init__()
  
	def _crossover(self, first_parent: _G, second_parent: _G) -> _Tup[_G, _G]:
		child1 = first_parent.clone()
		child2 = second_parent.clone()
		for pos, locuses in enumerate(child1.get_locuses()):
			shuffled = locuses.copy()
			self.rng().shuffle(shuffled)
			chromosome_copy_first = child1[pos].clone()
			chromosome_copy_second = child2[pos].clone()
			chromosome_copy_first[:] = child1[pos][shuffled]
			chromosome_copy_second[:] = child2[pos][shuffled]

			pivot_point = self.rng().choice(locuses)
			chromosome_copy_first[pivot_point:], chromosome_copy_second[pivot_point:] = (chromosome_copy_second[pivot_point:], chromosome_copy_first[pivot_point:])

			child1[pos][:] = chromosome_copy_first[shuffled] 
			child2[pos][:] = chromosome_copy_second[shuffled]

		return child1, child2