from SI2.utilities import *

from .types import *

if __name__ == "__main__":
	print(is_iterable_of(((2, 3), (2, 3), (2,3)), (_Tup[int, int], )))
 