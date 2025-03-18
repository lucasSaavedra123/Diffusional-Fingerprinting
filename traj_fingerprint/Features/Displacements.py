import numpy as np

from .Feature import Feature


class Displacements(Feature):
    def calculate(self, trajectory):
        return [np.mean(trajectory.displacements())]
