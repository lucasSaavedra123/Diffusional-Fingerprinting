from scipy.spatial import ConvexHull

from .Feature import Feature


class Area(Feature):
    @property
    def names(self):
        return ['Area']

    """
    Obtained from https://doi.org/10.21203/rs.3.rs-3716053/v1.
    Also, it is calculated in (Vogler, 2023).
    """
    def calculate(self, trajectory):
        return [ConvexHull(trajectory.raw_trajectory).volume]
