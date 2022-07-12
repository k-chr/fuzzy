import time
from typing import Any, Callable, Tuple
from .utilities import SpecialOptim
import numpy as np
from .ANFIS import ANFIS
from sklearn.metrics import accuracy_score

def measure_time(key, model: ANFIS, n_iter):
    start= time.time()
    model.train(True, True, False, True, n_iter=n_iter)
    end = time.time()
    print(f"FINISHED: {key}")
    return key, end - start, model

def train_using_ga(key, model: ANFIS, n_iter:int, **ga_kwargs):
    res = model.train(False, True, False, True, n_iter=n_iter, use_special= SpecialOptim.GA, **ga_kwargs)
    return key, res, model

def train_using_pso(key, model: ANFIS, n_iter:int, **pso_kwargs):
    res = model.train(False, True, False, True, n_iter=n_iter, use_special= SpecialOptim.PSO, **pso_kwargs)
    return key, res, model

def train(norm: Tuple[str, Callable[[np.ndarray, Any], np.ndarray]], train, varX, varY) -> Tuple[ANFIS, str, float]:
    X_train, y_train = train
    fis = ANFIS([varX, varY], X_train.T, y_train, operator_function=norm[1])
    start = time.time()
    fis.train(True, True, False, True, n_iter=100)
    end = time.time()
    elapsed_time = end - start
    fis.training_data = X_train.T
    fis.expected_labels = y_train
    return norm[0], fis, elapsed_time
    
def test(fis: ANFIS, test, name_of_op: str):
    X_test, y_test = test
    fis.training_data = X_test.T
    fis.expected_labels = [int(cell) for cell in (np.array(y_test).astype(dtype=int)*10)]
    y_pred = fis.anfis_estimate_labels(fis.premises, fis.op, fis.tsk).flatten()
    y_pred = list(map(round, y_pred.flatten() * 10))
    accuracy = accuracy_score(fis.expected_labels, y_pred)
    return name_of_op, fis, accuracy