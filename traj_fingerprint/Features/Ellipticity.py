import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from .Feature import Feature
from .Fingerprint_feat_gen import GetMax


class Ellipticity(Feature):
    @property
    def names(self):
        return ['Area']

    """
    Obtained from (Vogler, 2023).
    """
    def calculate(self, trajectory):
        A = np.array([trajectory.get_noisy_x(), trajectory.get_noisy_x()]).T
        B = np.array([[np.mean(trajectory.get_noisy_x())], [np.mean(trajectory.get_noisy_y())]]).T
        d_mean = (cdist(A,B)).mean()
        d_max = GetMax(trajectory.get_noisy_x(), trajectory.get_noisy_y())

        a_convex = ConvexHull(trajectory.raw_trajectory).volume
        
        a_ellipsis = np.pi * d_max * d_mean
        elli = a_convex / a_ellipsis
        return [elli]
