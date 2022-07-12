from .particle import Particle
from typing import Tuple as _Tup
import numpy as _np
from .. import PriorityQueue

class Swarm:
	def __init__(self, 
				 rng: _np.random.Generator, 
				 size: int,
				 dims: int,
				 constraints: _Tup[_Tup[int, int], ...],
				 omega: float,
				 alpha: float,
				 beta: float,
				 init_val: float =_np.inf) -> None:

		self.__size = size
		self.__particles = [Particle(rng, dims, constraints, init_val) for _ in range(size)]
		self.alpha = alpha
		self.omega = omega
		self.beta = beta
		self.__heap = PriorityQueue[Particle]()
		[self.__heap.push(particle, particle.best_value) for particle in self.__particles]		
  
	def update(self):
		[self.__heap.update(particle, particle.best_value) for particle in self.__particles]	
  
	def compute_velocity(self, particle: Particle, c1: _np.ndarray, c2: _np.ndarray):
		return self.omega * particle.velocity + c1 * self.alpha * (
      					particle.best_position - particle.position
           			) + c2 * self.beta * (
                  		self.best_particle.best_position - particle.position
                    )
  
	@property
	def best_particle(self): return self.__heap.peek(1)[0]
 
	def __getitem__(self, index):
		return self.__particles[index]

	def __iter__(self):
		self.__iter = 0
		return self

	def __next__(self):
		if self.__iter < self.__size:
			item = self.__getitem__(self.__iter)
			self.__iter += 1
			return item

		else: raise StopIteration()	
	