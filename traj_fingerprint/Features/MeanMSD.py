import numpy as np

from .Feature import Feature


class MeanMSD(Feature):
    def calculate(self, trajectory):
        return [np.mean(trajectory.info['msd'])]
