from typing import TypeVar as __TVar, List as _L, Generic as _G, Tuple as _Tup
import heapq as _hq
from . import Value as _V
from copy import deepcopy as _dcopy

_T = __TVar('_T')


class PriorityQueue(_G[_T]):

	def  __init__(self):
		self.__heap: _L[_Tup[_V, int, _T]] =[]
		self.count = 0
	
	@property
	def heap(self) -> _Tup[_Tup[_V, int, _T], ...]: return tuple(self.__heap)
 
	def push(self, item: _T, priority: _V):
		entry = (priority, self.count, item)
		_hq.heappush(self.__heap, entry)
		_hq._heapify_max(self.__heap)
		self.count += 1

	def pop(self) -> _T:
		(_, _, item) = _hq._heappop_max(self.__heap)
		self.count -= 1
		return item

	def is_empty(self):
		return len(self.__heap) == 0

	def update(self, item: _T, priority: _V):

		for index, (p, c, i) in enumerate(self.__heap):
			if i == item:
				if p >= priority:
					break
				del self.__heap[index]
				self.__heap.append((priority, c, item))
				_hq._heapify_max(self.__heap)
				break
		else:
			self.push(item, priority)
   
	def peek(self, n: int) -> _Tup[_T]:
		if n > self.count: n = self.count
		tuples = self.__heap[:n]
		return tuple(map(lambda x: _dcopy(x[2]), tuples))
