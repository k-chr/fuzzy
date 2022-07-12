from typing import Callable as _C, Dict as _D, Type as _T, Union as _U, List as _L, Tuple as _Tup
from . import *
import numpy as _np
import copy as _cp
from inspect import getfullargspec as _spec
from operator import itemgetter as _getter
import tqdm as _looper

_x_ops: _D[str, Crossover] = {"kpoint": KPointCrossover, 
		  "binrespect": RandomRespectfulCrossover, 
		  "shuffle": ShuffleCrossover}

_m_ops: _D[str, Mutation] = {"adjswap": AdjacentSwapMutation,
		  "randswap": RandomSwapMutation,
		  "randneg": RandomNegationMutation,
		  "sliceinv": SliceInversionMutation}

_s_ops: _D[str, Selection] = {"rank": RankSelection,
		  "tournament": TournamentSelection,
		  "roulette": RouletteWheelSelection}

def _get_x_op(name: str):
	return _x_ops.get(name, None)

def _get_m_op(name: str):
	return _m_ops.get(name, None)

def _get_s_op(name: str):
	return _s_ops.get(name, None)

def _make_fitness(fun: Fitness, minimize: bool= False, *args, **kwargs) -> _C[[Genome], Value]:
	def fitness(genome: Genome) -> Value:
		val = fun(_np.array(genome.decode_genetic_information()), *args, **kwargs)
		return val if not minimize else -val
	return fitness

_def_kwargs = {
	"k" : 1,
	"prob": 0.1,
	"m" : 1,
	"constr": [(0, 1)],
	"fun_args":[],
	"fun_kwargs": {},
}


def _get_init_spec(cl:_T):
	arg = _spec(cl).args
	try:
		arg.remove("self")
	except: ...
	return arg

def _init_op(op_class: _U[_T[Mutation], _T[Crossover], _T[Selection]], **kw):
	spec = _get_init_spec(op_class)
 
	if spec:
		args_getter = _getter(*spec)
		args = args_getter(kw)
  
		if type(args) not in [list, tuple]:
			args = [args]
   
	else: args=[]
 
	op = op_class(*args)
	return op

def optimize(fitness: Fitness,
			 minimize: bool =False,
			 num_of_iterations: int =100,
			 num_of_individuals: int =10,
			 elitism_ratio: float =0.2,
			 theta: float =1e-6,
			 k_best_criterion: float = 0.3,
			 sigma: float =None,
			 crossover: str ="kpoint",
			 mutation: str ="randneg",
			 selection: str ="rank",
			 elit_selection: str ="rank",
			 precision: ChromosomePrecision =ChromosomePrecision.SINGLE,
			 **op_kwargs) -> OptimizationResult:
	
	x_op_class = _get_x_op(crossover)
	m_op_class = _get_m_op(mutation)
	s_op_class = _get_s_op(selection)
	e_s_op_class = _get_s_op(elit_selection)
 
	assert all([x_op_class, m_op_class, s_op_class, e_s_op_class])

	kw = _cp.deepcopy(_def_kwargs) 
	kw.update(**op_kwargs)
	config = dict(function=fitness, num_of_iterations=num_of_iterations, num_of_individuals=num_of_individuals, elitism_ratio=elitism_ratio, theta=theta, k_best_criterion=k_best_criterion,
				sigma=sigma, crossover = crossover, mutation = mutation,selection =selection, elit_selection=elit_selection,precision=precision, minimize=minimize)
	config.update(**kw)
	
	fitness = _make_fitness(fitness, minimize, *(kw["fun_args"]), **(kw["fun_kwargs"]))
	fitness = measure_calls(fitness)
	kw["fitness_function"] = fitness 
	
	is_end: _C[[Genome, Genome], bool]

	if sigma is not None:
		is_end = lambda old_genome, genome: (
	  					abs(fitness(genome)-fitness(old_genome)) <= theta
		   		) or (
	  					((_np.asarray(genome.decode_genetic_information())-_np.asarray(old_genome.decode_genetic_information()))**2).sum() <= sigma
		 	)
	else: is_end = lambda old_genome, genome: (
	  			abs(fitness(genome)-fitness(old_genome)) <= theta
		   )

	mut_prob = kw.get("mutation_prob", 0.15)
	cross_prob = kw.get("crossover_prob", 0.25)

	params_optim = []
	epoch = 0
	best_value = 0.0
	
	x_op: Crossover = _init_op(x_op_class, **kw)
	m_op: Mutation = _init_op(m_op_class, **kw)
	s_op: Selection = _init_op(s_op_class, **kw)
	e_s_op: Selection = _init_op(e_s_op_class, **kw)
 
	rng = get_rng()	
	params_specification = kw['constr']
	population = [Genome(rng, precision, constraints=params_specification) for _ in range(num_of_individuals)]
	_loop = _looper.trange(num_of_iterations)
	elite_count = int(num_of_individuals * elitism_ratio) + num_of_individuals % 2
	normies_count = num_of_individuals - elite_count
	k_best = int(_np.round(num_of_individuals*k_best_criterion))
	for it in _loop:
		old_best_solution = e_s_op.select(population, 1)[0]
		new_population = []
		elite = e_s_op.select(population, elite_count)
		normies = s_op.select(population, normies_count)
  
		pairs: _L[_Tup[Genome, Genome]] = rng.choice(normies, (normies_count//2, 2), False).tolist()
  
		for pair in pairs:
			x_op_can_happen = rng.choice([True, False], p=[cross_prob, 1-cross_prob])
			if x_op_can_happen:
				pair = x_op.crossover(pair[0], pair[1])
	
			pair = (_mutate(mut_prob, m_op, rng, pair[0]), 
           			_mutate(mut_prob, m_op, rng, pair[1]))
	
			new_population += pair
   
		new_population += elite
		population = new_population
		epoch = it + 1
		if all([is_end(old_best_solution, solution) for solution in e_s_op.select(population, k_best)]): break
	best_solution = e_s_op.select(population, 1)[0]
	params_optim = _np.array(best_solution.decode_genetic_information())
	best_value = fitness(best_solution) * ((-1) ** int(minimize))
  
	return OptimizationResult(params_optim, epoch, best_value, fitness.calls, config)

def _mutate(mut_prob: float, m_op: Mutation, rng: _np.random.Generator, individual):
	m_op_can_happen = rng.choice([True, False], p=[mut_prob, 1-mut_prob])
	if m_op_can_happen:
		individual = m_op.mutate(individual)
	return individual
 
	
__all__ = ["optimize"]