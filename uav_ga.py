import random
import numpy as np
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, waypoints, sensors):
        self.waypoints = waypoints
        self.sensors = sensors
        self.fitness = -float("inf")
        self.metrics = {}

    def clone(self):
        new_routes = [
            route.copy() for route in self.waypoints
        ]

        new_sensors = self.sensors.copy()

        child = Individual(new_routes, new_sensors)
        child.fitness = self.fitness
        child.metrics = self.metrics.copy()

        return child


class UAVGeneticAlgorithm:
    def __init__(
        self,
        env,
        population_size=60,
        generations=100,
        waypoint_count=8,
        mutation_rate=0.15,
        crossover_rate=0.85
    ):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.waypoint_count = waypoint_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []

    def random_individual(self):
        routes = []

        for start in self.env.start_positions:
            route = [start]

            for _ in range(self.waypoint_count):
                x = random.randint(0, self.env.width - 1)
                y = random.randint(0, self.env.height - 1)
                route.append((x, y))

            routes.append(route)

        sensors = np.random.randint(
            0,
            2,
            size=(
                self.env.drone_count,
                self.waypoint_count + 1
            )
        )

        return Individual(routes, sensors)

    def line_cells(self, start, end):
        x1, y1 = start
        x2, y2 = end

        cells = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        error = dx - dy

        while True:
            cells.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * error

            if e2 > -dy:
                error -= dy
                x1 += sx

            if e2 < dx:
                error += dx
                y1 += sy

        return cells

    def evaluate(self, individual):
        visited = set()
        total_distance = 0
        hazard_count = 0
        blocked_count = 0

        for drone_index, route in enumerate(
            individual.waypoints
        ):
            for i in range(len(route) - 1):
                start = route[i]
                end = route[i + 1]

                dx = end[0] - start[0]
                dy = end[1] - start[1]

                total_distance += (
                    dx * dx + dy * dy
                ) ** 0.5

                cells = self.line_cells(start, end)

                for point in cells:
                    if self.env.is_blocked(point):
                        blocked_count += 1
                        continue

                    if individual.sensors[drone_index][i] == 1:
                        visited.add(point)

                        if self.env.is_hazard(point):
                            hazard_count += 1

        grid_area = self.env.width * self.env.height

        coverage = len(visited) / grid_area
        energy = total_distance / grid_area
        flight_time = total_distance / grid_area
        risk = hazard_count / grid_area
        constraint_penalty = blocked_count / grid_area

        fitness = (
            0.50 * coverage
            - 0.15 * energy
            - 0.10 * flight_time
            - 0.25 * risk
            - 0.50 * constraint_penalty
        )

        individual.fitness = fitness

        individual.metrics = {
            "coverage": coverage,
            "distance": total_distance,
            "hazard_cells": hazard_count,
            "blocked_violations": blocked_count
        }

        return fitness

    def initialize_population(self):
        self.population = [
            self.random_individual()
            for _ in range(self.population_size)
        ]

        for individual in self.population:
            self.evaluate(individual)

    def selection(self):
        candidates = random.sample(
            self.population,
            min(4, len(self.population))
        )

        return max(
            candidates,
            key=lambda item: item.fitness
        )

    def crossover(self, parent1, parent2):
        child = parent1.clone()

        if random.random() < self.crossover_rate:
            for drone in range(self.env.drone_count):
                if self.waypoint_count > 1:
                    point = random.randint(
                        1, self.waypoint_count - 1
                    )

                    child.waypoints[drone] = (
                        parent1.waypoints[drone][:point + 1]
                        + parent2.waypoints[drone][point + 1:]
                    )

            mask = np.random.rand(
                *parent1.sensors.shape
            ) < 0.5

            child.sensors = np.where(
                mask,
                parent1.sensors,
                parent2.sensors
            ).copy()

        return child

    def mutate(self, individual):
        for route in individual.waypoints:
            for i in range(1, len(route)):
                if random.random() < self.mutation_rate:
                    x, y = route[i]

                    x += random.randint(-5, 5)
                    y += random.randint(-5, 5)

                    x = max(0, min(self.env.width - 1, x))
                    y = max(0, min(self.env.height - 1, y))

                    route[i] = (x, y)

        for i in range(individual.sensors.shape[0]):
            for j in range(individual.sensors.shape[1]):
                if random.random() < self.mutation_rate:
                    individual.sensors[i, j] = (
                        1 - individual.sensors[i, j]
                    )

        return individual

    def run(self):
        self.initialize_population()

        best_history = []
        average_history = []

        best_overall = max(
            self.population,
            key=lambda item: item.fitness
        ).clone()

        for generation in range(self.generations):
            self.population.sort(
                key=lambda item: item.fitness,
                reverse=True
            )

            current_best = self.population[0]

            if current_best.fitness > best_overall.fitness:
                best_overall = current_best.clone()

            best_history.append(best_overall.fitness)

            average_history.append(
                np.mean([
                    item.fitness for item in self.population
                ])
            )

            new_population = [best_overall.clone()]

            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()

                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.evaluate(child)

                new_population.append(child)

            self.population = new_population

            if (generation + 1) % 10 == 0:
                print(
                    f"Generation {generation + 1}: "
                    f"Best Fitness = "
                    f"{best_overall.fitness:.5f}"
                )

        return best_overall, best_history, average_history

    def plot_convergence(self, best_history, average_history):
        plt.figure(figsize=(9, 5))

        plt.plot(best_history, label="Best Fitness")
        plt.plot(average_history, label="Average Fitness")

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Fitness Convergence")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_routes(self, individual):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            self.env.blocked,
            cmap="Greys",
            origin="lower",
            alpha=0.8
        )

        ax.imshow(
            self.env.hazards,
            cmap="Reds",
            origin="lower",
            alpha=0.35
        )

        colors = ["blue", "green", "orange"]

        for i, route in enumerate(individual.waypoints):
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]

            ax.plot(
                xs, ys,
                marker="o",
                color=colors[i],
                label=f"UAV {i + 1}"
            )

            ax.scatter(
                xs[0], ys[0],
                marker="s",
                color=colors[i],
                s=100
            )

        ax.set_xlim(0, self.env.width - 1)
        ax.set_ylim(0, self.env.height - 1)
        ax.set_title("Evolved UAV Routes")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()

    def plot_coverage(self, individual):
        coverage_map = np.zeros(
            (self.env.height, self.env.width)
        )

        for drone_index, route in enumerate(
            individual.waypoints
        ):
            for i in range(len(route) - 1):
                if individual.sensors[drone_index][i] == 1:
                    cells = self.line_cells(route[i], route[i + 1])

                    for x, y in cells:
                        if not self.env.is_blocked((x, y)):
                            coverage_map[y, x] += 1

        plt.figure(figsize=(8, 7))

        plt.imshow(
            coverage_map,
            cmap="YlOrRd",
            origin="lower"
        )

        plt.colorbar(label="Number of UAV passes")
        plt.title("UAV Coverage Heatmap")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        plt.tight_layout()
        plt.show()

    def random_search(self, trials=1000):
        best_score = -float("inf")

        for _ in range(trials):
            candidate = self.random_individual()
            score = self.evaluate(candidate)

            if score > best_score:
                best_score = score

        return best_score
