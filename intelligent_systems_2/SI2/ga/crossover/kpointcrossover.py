from . import Crossover as _Base
from .. import Genome as _G
from typing import Tuple as _Tup
import numpy as _np

class KPointCrossover(_Base):
    
	def __init__(self, k) -> None:
		super().__init__()
		self.k = k
  
	def _crossover(self, first_parent: _G, second_parent: _G) -> _Tup[_G, _G]:
		child1 = first_parent.clone()
		child2 = second_parent.clone()
		
		parents = [first_parent, second_parent]
		children = [child1, child2]
		first_id, second_id = 0, 1
		for pos, locuses in enumerate(child1.get_locuses()):
			chosen_locuses: _np.ndarray = self.rng().choice(locuses, self.k, False)
			chosen_locuses.sort()
			first_locus = 0
			
			for locus in chosen_locuses:
				children[first_id][pos][first_locus:locus], children[second_id][pos][first_locus:locus] = parents[0][pos][first_locus:locus], parents[1][pos][first_locus:locus]
				
				first_locus = locus
				first_id, second_id = second_id, first_id
			else:
				children[first_id][pos][locus:], children[second_id][pos][locus:] = parents[0][pos][locus:], parents[1][pos][locus:]
				first_id, second_id = second_id, first_id
		return child1, child2


			
