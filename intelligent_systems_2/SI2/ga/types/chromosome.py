import struct as _struct
from typing import Tuple as _Tup, Union as _U, Sequence as _I
import numpy as _np
from .. import Value as _V
from enum import Enum as _E
import operator as _op

class ChromosomePrecision(_E):
	HALF = "!e"
	SINGLE = "!f"
	DOUBLE = "!d"

	def get_locuses(self) -> range:
		start, stop = 0, 1
		if self is ChromosomePrecision.SINGLE:
			start, stop = 2, 32
		elif self is ChromosomePrecision.HALF:
			start, stop = 2, 16
		elif self is ChromosomePrecision.DOUBLE:
			start, stop = 2, 64
   
		return range(start, stop)

	def get_constraints(self) -> _Tup[float, float]:
		r = self.get_locuses()
		start, stop = r.start, r.stop

		_min = 0

		_bits = _np.array([0] * start + [1] * (stop - start), dtype=_np.uint8)
		_bytes = _np.packbits(_bits)
		_max, = _struct.unpack(self.value, bytearray(_bytes))

		return _min, _max
    	

class Chromosome:
	def __init__(self, value_range: _Tup[_V, _V], prob: float, precision: ChromosomePrecision =ChromosomePrecision.SINGLE) -> None:
		assert prob <= 1.0 and prob >= 0.0 and len(value_range) == 2
		_, self.__helper_ub = precision.get_constraints()
		self.__lb, self.__ub = min(value_range), max(value_range)
		self.__precision = precision
		self.__prob = prob
		_bytes = [b for b in bytearray(_struct.pack(self.__precision.value, self.__prob * self.__helper_ub))]
		self.__bits: _np.ndarray = _np.unpackbits(_np.array(_bytes, dtype=_np.uint8))

		self.__mutable_locuses = list(self.__precision.get_locuses())
		self.__overflow_guard = self.__mutable_locuses[0]
  
		self.__mutated = False
		self.__value = (self.__lb + self.__prob * (self.__ub - self.__lb))
		self.__is_val_computed = True

	@property
	def locuses(self) -> _np.ndarray: return _np.array(self.__mutable_locuses) - self.__overflow_guard
 
	@property
	def prob(self): 
		if self.__mutated:
			_bytes = _np.packbits(self.__bits)
			self.__prob, = _struct.unpack(self.__precision.value, bytearray(_bytes)) 
			self.__prob /= self.__helper_ub
			self.__mutated = False
		return self.__prob
 
	@property
	def value(self) -> _V:
		if not self.__is_val_computed: 
			self.__value = (self.__lb + self.prob * (self.__ub - self.__lb))
			self.__is_val_computed = True
		return self.__value

	def __map_getter(self, getter: _U[int, slice, _I, _np.ndarray])-> _Tup[bool, _U[int, slice, _I, _np.ndarray]]:
		is_single = False
		if isinstance(getter, int) or isinstance(getter, _np.ndarray) or isinstance(getter, list):
			is_single = True
			getter = getter + self.__overflow_guard
		elif isinstance(getter, slice):
			is_single = True
			getter = slice(getter.start+self.__overflow_guard if getter.start else self.__overflow_guard, getter.stop+self.__overflow_guard if getter.stop else None, getter.step)
		elif isinstance(getter, _I):
			getter = [_i + self.__overflow_guard for _i in getter]

		return is_single, getter
	
	def __getitem__(self, getter: _U[int, slice, _I, _np.ndarray]):
		is_single, getter = self.__map_getter(getter)
  
		callback = _op.itemgetter(*getter) if not is_single else _op.itemgetter(getter)
		bits = callback(self.__bits)
		return bits
   
	def __setitem__(self, getter: _U[int, slice, _I, _np.ndarray], value: _U[_np.ndarray, _V]):
		_, getter = self.__map_getter(getter)
		copy = self.__bits.copy()

		try:
			self.__bits[getter] = value
			
			self.__mutated = True
			self.__is_val_computed = False
		except:
			print("Exception")
			self.__bits = copy
	
	def clone(self):
		return Chromosome((self.__ub, self.__lb), self.prob, precision=self.__precision)
		
	def check_and_fix_overflow(self):
		if self.__bits[self.__overflow_guard] and sum(self.__bits[self.__overflow_guard + 1:]) > 0:
			self.__bits[self.__overflow_guard + 1:] = 0
			self.__mutated = True
			self.__is_val_computed = False
   
	def __repr__(self) -> str:
		return "".join(map(lambda gene: str(gene), self.__bits.tolist()))