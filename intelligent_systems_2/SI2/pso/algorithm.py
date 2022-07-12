from . import Fitness, OptimizationResult, Particle, Value, Swarm, get_rng, measure_calls
import numpy as _np
from typing import Callable as _C, Optional as _O
import tqdm as _looper
import copy as _cp

def _make_fitness(fun: Fitness, minimize: bool= False, *args, **kwargs) -> _C[[Particle], Value]:
	def fitness(particle: Particle) -> Value:
		val = fun((particle.position), *args, **kwargs)
		return val if not minimize else -val
	return fitness

_def_kwargs = {
	"constr": [(0, 1)],
	"fun_args":[],
	"fun_kwargs": {},
}

def optimize(fitness: Fitness,
			 minimize: bool =False,
			 num_of_iterations: int =100,
			 num_of_particles: int =10,
			 dims: int =1,
			 omega: float =0.5,
			 alpha: float =0.5,
			 beta: float =0.5,
			 theta: float =1e-6,
			 sigma: _O[float] =None,
			 **op_kwargs) -> OptimizationResult:

	kw = _cp.deepcopy(_def_kwargs) 
	kw.update(**op_kwargs)
	config = dict(function=fitness, num_of_iterations=num_of_iterations, num_of_particles=num_of_particles, dims=dims, omega=omega, alpha=alpha, beta=beta, theta=theta,
				sigma=sigma, minimize=minimize)
	config.update(**kw)
	# record: bool = kw.get("record_gif", False)
	# if record: raw = fitness
	fitness = measure_calls(_make_fitness(fitness, minimize, *(kw["fun_args"]), **(kw["fun_kwargs"])))
	kw["fitness_function"] = fitness 
 
	rng = get_rng()
	params_specification = kw['constr']
	init_val = -_np.inf 	
 
	swarm = Swarm(rng, num_of_particles, dims, params_specification, omega, alpha, beta, init_val=init_val)
 
	def __update_particle(particle: Particle):
		val = fitness(particle)
		if val > particle.best_value:
			particle.best_value = val
			particle.best_position = particle.position.copy()
   
	[__update_particle(particle) for particle in swarm]
	swarm.update()

	# if record:
	# 	import matplotlib.pyplot as plt
	# 	from matplotlib.animation import FuncAnimation
	# 	bounds = swarm.best_particle.domain
	# 	x, y = _np.array(_np.meshgrid(_np.linspace(*bounds[0].tolist(),1000), _np.linspace(*bounds[1].tolist(),1000)))
	# 	z = raw([x, y])
	# 	x_min = x.ravel()[z.argmin()]
	# 	y_min = y.ravel()[z.argmin()]
	# 	fig, ax = plt.subplots(figsize=(8,6))
	# 	fig.set_tight_layout(True)
	# 	img = ax.imshow(z, extent=[*bounds[0].tolist(), *bounds[1].tolist()], origin='lower', cmap='viridis', alpha=0.5)
	# 	fig.colorbar(img, ax=ax)
	# 	ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
	# 	contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
	# 	ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
	# 	pbest = _np.array([particle.best_position for particle in swarm])
	# 	X = _np.array([particle.position for particle in swarm])
	# 	V = _np.array([particle.velocity for particle in swarm])
	# 	gbest = swarm.best_particle.best_position
	# 	pbest_plot = ax.scatter(pbest[:, 0], pbest[:, 1], marker='o', color='black', alpha=0.5)
	# 	p_plot = ax.scatter(X[:, 0], X[:, 1], marker='o', color='blue', alpha=0.5)
	# 	p_arrow = ax.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
	# 	gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=200, color='red', alpha=0.8)
	# 	ax.set_xlim(bounds[0].tolist())
	# 	ax.set_ylim(bounds[1].tolist())
	
	is_end: _C[[Particle, Particle], bool] 
 
	if sigma is not None:
		is_end = lambda old_particle, particle: (
	  					abs(fitness(particle)-fitness(old_particle)) <= theta
		   		) or (
	  					((particle.position-old_particle.position)**2).sum() <= sigma
		 	)
	else: is_end = lambda old_particle, particle: (
	  			abs(fitness(particle)-fitness(old_particle)) <= theta
		   )

	params_optim = []
	epoch = 0
	best_value = 0.0

	_loop = _looper.trange(num_of_iterations)

	for it in _loop:
		previous_best = swarm.best_particle

		c_1 = rng.uniform(size=(num_of_particles, dims))
		c_2 = rng.uniform(size=(num_of_particles, dims))

		for idx, particle in enumerate(swarm):

			particle.velocity = swarm.compute_velocity(particle, c_1[idx], c_2[idx])
			particle.position += particle.velocity
			__update_particle(particle)
	
		swarm.update()
		epoch = it + 1
		if is_end(previous_best, swarm.best_particle): break

	# else:
	# 	def u(it):
	# 		global epoch
	# 		previous_best = swarm.best_particle
	# 		if it %10 == 0:
	# 			print(f"current best: {previous_best.best_value}")
	# 		c_1 = rng.uniform(size=(num_of_particles, dims))
	# 		c_2 = rng.uniform(size=(num_of_particles, dims))

	# 		for idx, particle in enumerate(swarm):

	# 			particle.velocity = swarm.compute_velocity(particle, c_1[idx], c_2[idx])
	# 			particle.position += particle.velocity
	# 			__update_particle(particle)
	
	# 		swarm.update()
	# 		epoch = it + 1
	# 	def animate(i):
	# 		"Steps of PSO: algorithm update and show in plot"
	# 		title = 'Iteration {:02d}'.format(i)
	# 		# Update params
	# 		u(i)
	# 		# Set picture
	# 		pbest = _np.array([particle.best_position for particle in swarm])
	# 		X = _np.array([particle.position for particle in swarm])
	# 		V = _np.array([particle.velocity for particle in swarm])
	# 		gbest = swarm.best_particle.best_position
	# 		ax.set_title(title)
	# 		pbest_plot.set_offsets(pbest)
	# 		p_plot.set_offsets(X)
	# 		p_arrow.set_offsets(X)
	# 		p_arrow.set_UVC(V[:, 0], V[:, 1])
	# 		gbest_plot.set_offsets(gbest)
	# 		return ax, pbest_plot, p_plot, p_arrow, gbest_plot

	# 	anim = FuncAnimation(fig, animate, frames=list(range(1,num_of_iterations+1)), interval=500, blit=False, repeat=True)
	# 	anim.save("PSO_HAHA.gif", dpi=120, writer="imagemagick")
	best_solution = swarm.best_particle
	params_optim = _np.array(best_solution.best_position)
	best_value = best_solution.best_value * (-1 ** (int(minimize)))
	return OptimizationResult(params_optim, epoch, best_value, fitness.calls, config)

__all__ = ["optimize"]