import numpy as np

def __productN(args, op):
    return np.product(args, axis=0)

def __zadehN(args, op):
    return np.fmin(*args)

def __lukasiewiczN(args, op):
    return np.fmax(np.zeros_like(args[0]), np.sum(args, axis=0)-1)

def __drasticN(args, op):
    z = np.zeros_like(args[0])
    
    z[args[0] == 1] = args[1][args[0] == 1]
    z[args[1] == 1] = args[0][args[1] == 1]
    return z

def __einsteinN(args, op):
    a = args[0]
    b = args[1]
    numerator = __productN(args, op)
    return np.divide(numerator, 2*np.ones_like(a) - a - b + numerator)

def __fodorN(args, op):
    s = np.sum(args, axis=0)
    a = args[0]
    b = args[1]
    fmin = np.fmin(a, b)
    return np.where(s > 1, fmin, 0)

def __chrustekN(args, op=None):
    a = np.arctan((args.prod(axis=0)+1e-8))
    a =  (a-a.min())/(a.max() - a.min())
    return a

TNORMS = {"Algebraic T-Norm": __productN, 
          "Zadeh T-Norm": __zadehN, 
          "Łukasiewicz T-Norm": __lukasiewiczN,
          "Drastic T-Norm": __drasticN,
          "Einstein T-Norm": __einsteinN,
          "Fodor T-Norm": __fodorN,
          "Chrustowski T-Norm": __chrustekN}

__all__ = ["TNORMS"]