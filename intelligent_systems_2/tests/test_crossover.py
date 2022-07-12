from SI2.ga import Genome, ChromosomePrecision, KPointCrossover, Chromosome, ShuffleCrossover, RandomRespectfulCrossover
import unittest
import numpy as np
import SI2.ga as ga


class TestCrossover(unittest.TestCase):

	def setUp(self) -> None:
		self.__rng = np.random.default_rng(2137)
		ga.get_rng = lambda: self.__rng
  
	def test_1_point_crossover(self):
		parent1 = Genome(self.__rng, precision=ChromosomePrecision.HALF, chromosomes=[Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF)])
		parent2 = Genome(self.__rng, chromosomes=[Chromosome((-2, 2), 0.9, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.1, precision=ChromosomePrecision.HALF)])
		print()
		print(parent1, "->", parent1.decode_genetic_information())
		print(parent2, "->", parent2.decode_genetic_information())
		
		mutator = KPointCrossover(1)
		children = mutator.crossover(parent1, parent2)

		print(children[0], "->", children[0].decode_genetic_information())
		print(children[1], "->", children[1].decode_genetic_information())
  
	def test_shuffle_crossover(self):
		parent1 = Genome(self.__rng, precision=ChromosomePrecision.HALF, chromosomes=[Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF)])
		parent2 = Genome(self.__rng, chromosomes=[Chromosome((-2, 2), 0.9, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.1, precision=ChromosomePrecision.HALF)])
		print()
		print(parent1, "->", parent1.decode_genetic_information())
		print(parent2, "->", parent2.decode_genetic_information())
		mutator = ShuffleCrossover()
		children = mutator.crossover(parent1, parent2)

		print(children[0], "->", children[0].decode_genetic_information())
		print(children[1], "->", children[1].decode_genetic_information())
  
	def test_random_respectful_crossover(self):
		parent1 = Genome(self.__rng, precision=ChromosomePrecision.HALF, chromosomes=[Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.5, precision=ChromosomePrecision.HALF)])
		parent2 = Genome(self.__rng, chromosomes=[Chromosome((-2, 2), 0.9, precision=ChromosomePrecision.HALF), Chromosome((-2, 2), 0.1, precision=ChromosomePrecision.HALF)])
		print()
		print(parent1, "->", parent1.decode_genetic_information())
		print(parent2, "->", parent2.decode_genetic_information())
		mutator = RandomRespectfulCrossover()
		children = mutator.crossover(parent1, parent2)
		print(children[0], "->", children[0].decode_genetic_information())
		print(children[1], "->", children[1].decode_genetic_information())
