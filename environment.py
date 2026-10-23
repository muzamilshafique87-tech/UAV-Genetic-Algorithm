import random
import numpy as np


class DisasterGrid:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.drone_count = 3
        self.start_positions = [(2, 2), (2, 47), (47, 2)]

        self.blocked = np.zeros((height, width), dtype=bool)
        self.hazards = np.zeros((height, width), dtype=bool)
        self.priority = np.zeros((height, width), dtype=bool)

        self._place_cells(self.blocked, 55)
        self._place_cells(self.hazards, 45)
        self._place_cells(self.priority, 70)

        for x, y in self.start_positions:
            self.blocked[y, x] = False
            self.hazards[y, x] = False
            self.priority[y, x] = True

        self.priority[self.blocked] = False
        self.hazards[self.blocked] = False

    def _place_cells(self, grid, count):
        for _ in range(count):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            grid[y, x] = True

    def is_blocked(self, point):
        x, y = point
        return (
            x < 0 or x >= self.width
            or y < 0 or y >= self.height
            or self.blocked[y, x]
        )

    def is_hazard(self, point):
        x, y = point
        return self.hazards[y, x]
