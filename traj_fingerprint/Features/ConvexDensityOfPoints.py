from scipy.spatial import ConvexHull

from .Feature import Feature


class ConvexDensityOfPoints(Feature):
    @property
    def names(self):
        return [r'$D$']

    """
    Obtained from (Vogler, 2023).
    """
    def calculate(self, trajectory):
        n_rho = trajectory.length
        a_convex = ConvexHull(trajectory.raw_trajectory).volume
        density = n_rho / a_convex
        return [density]
