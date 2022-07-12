from SI2.ga import algorithm as ga_alg
from SI2.pso import algorithm as pso_alg
import unittest
import numpy as np
import SI2.ga as ga
import SI2.pso as pso
from scipy.optimize import minimize

def fun_to_optimize_ga(ndarray):
	"""# -x^2 + y^2 - 2xy"""
	x, y = ndarray[0], ndarray[1]
	return (-x**2 + y**2 - 2*x*y)

def fun_to_optimize_pso(ndarray):
	"""# (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2"""
	x, y = ndarray[0], ndarray[1]
	return (1.5 - x - x * y)**2 + (2.25 - x + x * (y ** 2))**2 + (2.625 - x + x * (y ** 3))**2


class TestAlgorithm(unittest.TestCase):
	
	def setUp(self) -> None:
		self.__rng = np.random.default_rng(123)
		ga.get_rng = lambda: self.__rng
		pso.get_rng = lambda: self.__rng
  
	def test_genetic_algorithm_on_two_arg_function(self):
		optim_result = ga_alg.optimize(fun_to_optimize_ga, False, 100, 30, selection="roulette",sigma=1e-25, theta=1e-25, constr=[(-8, 8), (-8, 8)])
		print(optim_result)
		print(minimize(lambda x: -fun_to_optimize_ga(x), [3, 3], bounds=[(-8, 8), (-8, 8)]))
  
	def test_pso_algorithm_on_two_arg_function(self):
		optim_result = pso_alg.optimize(fun_to_optimize_pso, True, 50, 20, dims=2, omega=0.8, beta=0.1, alpha=0.1,  sigma=1e-25, theta=1e-25, constr=[(-4.5, 4.5), (-4.5, 4.5)], record_gif=True)
		print(optim_result)
		print(minimize(fun_to_optimize_pso, [0, 0], bounds=[(-4.5, 4.5), (-4.5, 4.5)]))	
