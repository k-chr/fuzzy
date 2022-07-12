import numpy as np

TESTS = [False]

def __zadeh_coN(args, op):
    return np.fmax(*args)

def __probabilistic_coN(args, op):
	return (1 - np.product(1 - args, axis=0))

def __lukasiewicz_coN(args, op):
    return np.fmin(np.ones_like(args[0]), np.sum(args, axis=0))

def __fodor_coN(args, op): 
    s = np.sum(args, axis=0)
    a = args[0]
    b = args[1]
    fmax = np.fmax(a, b)
    return np.where(s < 1, fmax, 1)

def __einstein_coN(args, op):
    return np.divide(np.sum(args, axis=0), np.product(args, axis=0) + 1)

def __drastic_coN(args, op):
    z = np.ones_like(args[0])
    
    z[args[0] == 0] = args[1][args[0] == 0]
    z[args[1] == 0] = args[0][args[1] == 0]
    return z

def __chrustek_coN(args, op=None):
    a = np.arctan((1 - np.product(1 - args, axis=0)) + 1e-8)
    a =  (a-a.min())/(a.max() - a.min())
    return a

SNORMS = {"Probabilistic T-CoNorm": __probabilistic_coN, 
          "Zadeh T-CoNorm": __zadeh_coN, 
          "Łukasiewicz T-CoNorm": __lukasiewicz_coN,
          "Drastic T-CoNorm": __drastic_coN,
          "Einstein T-CoNorm": __einstein_coN,
          "Fodor T-CoNorm": __fodor_coN,
          "Chrustowski T-CoNorm": __chrustek_coN}

__all__ = ['SNORMS']