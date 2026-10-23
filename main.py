import random
import numpy as np

from environment import DisasterGrid
from uav_ga import UAVGeneticAlgorithm


random.seed(42)
np.random.seed(42)

env = DisasterGrid(width=50, height=50)

ga = UAVGeneticAlgorithm(
    env=env,
    population_size=60,
    generations=100,
    waypoint_count=8,
    mutation_rate=0.15,
    crossover_rate=0.85
)

best, best_history, average_history = ga.run()

print("\nBest Fitness:", best.fitness)
print("Best Solution Metrics:", best.metrics)

ga.plot_convergence(best_history, average_history)
ga.plot_routes(best)
ga.plot_coverage(best)

random_score = ga.random_search(trials=1000)

print("\nRandom Search Fitness:", random_score)
print("Genetic Algorithm Fitness:", best.fitness)

if best.fitness > random_score:
    print("Result: Genetic Algorithm performed better.")
elif best.fitness == random_score:
    print("Result: Both methods achieved equal fitness.")
else:
    print("Result: Random Search performed better.")
