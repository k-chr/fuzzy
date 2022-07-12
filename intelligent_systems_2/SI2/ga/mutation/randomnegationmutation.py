from . import Mutation as _Base
from .. import Genome as _G

class RandomNegationMutation(_Base):

	def __init__(self, prob: float, m: int) -> None:
		super().__init__(prob)
		self.__k = m
  
	def _mutate(self, genome: _G) -> _G:
		clone = genome.clone()
		for pos, locuses in enumerate(genome.get_locuses()):
			mutate = self.rng().choice([True, False], p=[self.prob, 1-self.prob])
			if mutate:
				chosen_locuses = self.rng().choice(locuses, self.__k, replace=False)
				clone[pos][chosen_locuses] = 1 - clone[pos][chosen_locuses]
		return clone