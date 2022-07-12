import numpy as _np
from typing import Union as _U, Tuple as _Tup, Type as _T, Protocol as _Protocol, Dict as _D, Any as _Any, Callable as _C, TypeVar as __TVar, Generic as _G
from collections.abc import Iterable as _I
from dataclasses import dataclass as _dataclass
from numpy.random import default_rng as _d_rng, Generator as _Rng
import sys as _sys
import time as _t
from enum import Enum as _E

class SpecialOptim(_E):
    NONE = 0 
    PSO = 1
    GA = 2

Value = _U[int, float, _np.float32, _np.float64, _np.int32, _np.int16, _np.int8, _np.int64, _np.intc, _np.float16]

@_dataclass
class OptimizationResult:
	x: _np.ndarray
	nit: int
	fun: Value
	nfev: int
	config: _D[str, _Any]

def get_rng() -> _Rng:
	this = _sys.modules[__name__]
	if not hasattr(this, "gen"): 
		seed = _t.time_ns()//100
		print("Init rng with seed: ", seed)
  
		setattr(this, "gen", _d_rng(seed))
	return getattr(this, "gen")


class Fitness(_Protocol):
	def __call__(self, arr: _np.ndarray,*args, **kwargs) -> Value:...

_In = __TVar('_In')
_Out = __TVar('_Out')

class _MeasureCalls(_G[_In, _Out]):
		def __init__(self, target: _C[[_In], _Out]) -> None:
			self.__foo = target
			self.__calls = 0

		def __call__(self, *args: _In, **_) -> _Out:
			self.__calls += 1
			return self.__foo(*args)

		@property
		def calls(self): return self.__calls

def measure_calls(target: _C[[_In], _Out]):
    return _MeasureCalls(target)

def _get_inner(_type: _T): return tuple(t for t in getattr(_type, '__args__') if t is not ...) if hasattr(_type, '__args__') else _type
def _has_inner(_type: _T): return hasattr(_type, '__args__')

def is_iterable_of(iterable: _I, of:_Tup[_T,...]):
	if not isinstance(iterable, _I): return False
	assert len(of) == 1 or (len(of) > 1 and len(of) == len(iterable))
	if len(of) == 1: of *= len(iterable)	
	return all(list(map(_helper, zip(iterable, of))))

def _helper(item):
	obj, _type = item
	if not (isinstance(obj, _I) and _has_inner(_type)):
		if  hasattr(_type, '__origin__'):
			if getattr(_type, '__origin__') == _U: _type = _get_inner(_type)
		return isinstance(obj, _type)
	else:
		return is_iterable_of(obj, _get_inner(_type))

from .priorityqueue import *

__all__ = [
	"PriorityQueue", "Value", "is_iterable_of", "get_rng", "Fitness", "OptimizationResult", "measure_calls", "SpecialOptim"
]