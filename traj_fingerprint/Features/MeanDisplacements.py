import numpy as np

from .Feature import Feature


class MeanDisplacements(Feature):
    def calculate(self, trajectory):
        return [np.mean(trajectory.displacements())]
