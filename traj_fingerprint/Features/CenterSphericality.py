import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from .Feature import Feature


class CenterSphericality(Feature):
    @property
    def names(self):
        return [r'$C_{Spher}$']

    """
    Obtained from (Vogler, 2023).
    """
    def calculate(self, trajectory):
        A = np.array([trajectory.get_noisy_x(), trajectory.get_noisy_x()]).T
        B = np.array([[np.mean(trajectory.get_noisy_x())], [np.mean(trajectory.get_noisy_y())]]).T
        d_mean = (cdist(A,B)).mean()

        a_circle = np.pi * (d_mean**2)
        a_convex = ConvexHull(trajectory.raw_trajectory).volume
        
        center_sphericality = a_convex / a_circle
        return [center_sphericality]
