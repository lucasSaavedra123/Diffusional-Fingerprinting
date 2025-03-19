import numpy as np

from .Feature import Feature


class Velocity(Feature):
    @property
    def names(self):
        return ['MeanVelocity']

    """
    Obtained from https://doi.org/10.3390/receptors4010006
    """
    def calculate(self, trajectory):
        velocities = trajectory.displacements()/np.diff(trajectory.get_time())
        return [np.mean(velocities)]