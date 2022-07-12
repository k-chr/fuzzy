from __future__ import annotations
from typing import Any, Dict, List, Union, Tuple, Union

import numpy as np
from sklearn import svm

def _get_count(x: Union[np.ndarray, List[Any], Any]):
    if type(x) == list:
        return len(x)
    elif type(x) == np.ndarray:
        return x.shape[0]
    else:
        return 1

class FSVM(object):
    def __init__(self, C: float =1.0, kernel: str ='rbf', degree: int =3, 
                 gamma: Union[str, float] ='scale', max_iter: int =-1, 
                 seed: int =12) -> None:
        super().__init__()
        self.__C = C
        self.__kernel = kernel
        self.__gamma = gamma
        self.__seed = seed
        self.__svms: Dict[Tuple[Any, Any], svm.SVC] ={}
        self.__classes: List[Any] =[]
        self.__tuples: Dict[Any, List[Tuple[Any, Any]]] ={}
        self.__max_iter = max_iter
        self.__degree = degree

    def __memberships_i_j(self, decisions_i_j: np.ndarray) -> np.ndarray:
        return np.where(decisions_i_j >= 1, 1, decisions_i_j)   
    
    def __generate_membership_functions(self, x):
        memberships: np.ndarray =np.zeros((_get_count(x), len(self.__classes)))
        for index, _class in enumerate(self.__classes):
            tups = self.__tuples[_class]
            count = len(tups)
            decisions_arr: np.ndarray =np.zeros((_get_count(x), count))
            for _tup_idx, tup in enumerate(tups):
                __svm_decision: np.ndarray =self.__svms[tup].decision_function(x)
                decisions_arr[:, _tup_idx] = __svm_decision
            __memberships_i: np.ndarray =self.__memberships_i_j(decisions_arr)
            memberships[:, index] = np.min(__memberships_i, axis=1)
        return memberships

    def fit(self, X, Y):
        self.__classes = sorted(list(set(Y)))
        self.__svms = {}
        self.__tuples = {}
        for _class in self.__classes:
            for __class in  self.__classes:
                if _class != __class:
                    if _class not in self.__tuples or len(self.__tuples[_class]) < 1:
                        self.__tuples[_class] = []
                    self.__tuples[_class].append((_class, __class))
                    _x = np.vstack([X[Y == _class], X[Y == __class]]) 
                    _y = np.hstack([Y[Y == _class], Y[Y == __class]])
                    _y = np.where(_y == _class, 1, -1) 
                    self.__svms[(_class, __class)] = svm.SVC(C=self.__C, gamma=self.__gamma,
                                                             max_iter=self.__max_iter,
                                                             kernel=self.__kernel, degree=self.__degree,
                                                             random_state=self.__seed)
                    __svm = self.__svms[(_class, __class)]
                    __svm.fit(_x, _y)
        return self

    def decision_function(self, X):
        return self.__generate_membership_functions(X)
    
    def predict(self, X):
        memberships = self.__generate_membership_functions(X)
        return np.argmax(memberships, axis=1)
