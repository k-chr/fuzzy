from . import Mutation as _Base
from .. import Genome as _G


class AdjacentSwapMutation(_Base):

	def __init__(self, prob: float, m: int) -> None:
		super().__init__(prob)
		self.__k = m
  
	def _mutate(self, genome: _G) -> _G:
		clone = genome.clone()
		assert self.__k <= len(genome[0][:]) // 2
		for pos, locuses in enumerate(genome.get_locuses()):
			mutate = self.rng().choice([True, False], p=[self.prob, 1-self.prob])
			if mutate:
				chosen_locuses = self.rng().choice(locuses[:-1], size=self.__k, replace=False)
				for item in chosen_locuses:
					pair = [item, item+1]
					clone[pos][pair] = clone[pos][pair[::-1]]
		return clone