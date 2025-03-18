import numpy as np

from .Feature import Feature


class RadialDistances(Feature):
    """
    Obtained from https://doi.org/10.3390/receptors4010006
    """
    def calculate(self, trajectory):
        x,y = trajectory.get_noisy_x(), trajectory.get_noisy_y()
        t = np.array([x,y]).T
        radial_distances = np.linalg.norm(t-t[0])[1:]
        return [np.mean(radial_distances), np.max(radial_distances), np.min(radial_distances), np.max(radial_distances)-np.min(radial_distances)]
