from . import Mutation as _Base
from .. import Genome as _G
import numpy as _np

class SliceInversionMutation(_Base):

	def __init__(self, prob: float) -> None:
		super().__init__(prob)
  
	def _mutate(self, genome: _G) -> _G:
		clone = genome.clone()
		for pos, locuses in enumerate(genome.get_locuses()):
			mutate = self.rng().choice([True, False], p=[self.prob, 1-self.prob])
			if mutate:
				chosen_locuses: _np.ndarray = self.rng().choice(locuses, size=2, replace=False)
				chosen_locuses.sort()
				clone[pos][chosen_locuses[0] : chosen_locuses[1]] = (clone[pos][chosen_locuses[0]:chosen_locuses[1]])[::-1]
		return clone