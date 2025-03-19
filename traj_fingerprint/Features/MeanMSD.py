import numpy as np

from .Feature import Feature


class MeanMSD(Feature):
    @property
    def names(self):
        return ['MeanMSD']

    def calculate(self, trajectory):
        return [np.mean(trajectory.info['msd'])]
