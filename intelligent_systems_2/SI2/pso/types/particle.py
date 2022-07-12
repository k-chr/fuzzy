from __future__ import annotations
from typing import Tuple as _Tup
import numpy as _np

from .. import Value as _V, is_iterable_of as _type_check


class Particle:
	def __init__(self, rng: _np.random.Generator, dims: int, domain: _Tup[_Tup[_V, _V],...], init_val: float =_np.inf) -> None:
		self.__rng = rng
  
		assert _type_check(domain, (_Tup[_V, _V], ))
		assert  (len(domain) > 1 and len(domain) == dims) or len(domain) == 1
		
		self.__domain = _np.asarray(domain if len(domain) == dims else domain * dims)
		self.__domain.sort(axis=-1)
		position_range = (self.__domain[:,1] - self.__domain[:,0])
		velocity_range = 2 * _np.abs(position_range)
		
		self.__position: _np.ndarray =self.__domain[:,0] + self.__rng.uniform(size=dims) * position_range
		self.__velocity: _np.ndarray = -_np.abs(position_range) + self.__rng.random(size=dims) * velocity_range #self.__rng.standard_normal(dims) * 0.1
		self.__best_val = init_val
		self.__best_position = self.__position.copy()
  
	@property
	def position(self): return self.__position
 
	@property
	def best_value(self): return self.__best_val
 
	@property
	def best_position(self): return self.__best_position
 
	@property
	def velocity(self): return self.__velocity 
 
	@position.setter
	def position(self, value: _np.ndarray):
		self.__position = _np.clip(value, self.__domain[:,0], self.__domain[:,1])
  
	@best_value.setter
	def best_value(self, value:float):
		self.__best_val = value

	@best_position.setter
	def best_position(self, value: _np.ndarray): 
		self.__best_position = value
  
	@velocity.setter
	def velocity(self, value):
		self.__velocity = value
	
	@property
	def domain(self): return self.__domain
	
	def __str__(self) -> str:
		return f"Particle at: {hex(id(self))},\n\tcurrent position: {self.position},\n\tcurrent velocity: {self.velocity},\n\tbest position: {self.best_position},\n\tbest value: {self.best_value}"

if __name__ == "__main__":
	particle = Particle(_np.random.default_rng(2137), 8, ((-3, 3),))
	print(particle)
	particle.position += particle.velocity
	print(particle)
