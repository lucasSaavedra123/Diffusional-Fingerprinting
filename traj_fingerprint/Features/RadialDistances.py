import numpy as np

from .Feature import Feature


class RadialDistances(Feature):
    @property
    def names(self):
        return [
            'RadialDistMean',
            'RadialDistMax',
            'RadialDistMin',
            'RadialDistMaxMin',
        ]

    """
    Obtained from https://doi.org/10.3390/receptors4010006
    """
    def calculate(self, trajectory):
        trajectory = trajectory.raw_trajectory
        radial_distances = np.linalg.norm(trajectory-trajectory[0], axis=1)[1:]
        return [np.mean(radial_distances), np.max(radial_distances), np.min(radial_distances), np.max(radial_distances)-np.min(radial_distances)]
