import numpy as np

from .Feature import Feature


class Directionality(Feature):
    """
    Obtained from https://doi.org/10.21203/rs.3.rs-3716053/v1
    and https://doi.org/10.1039/D3NA00188A
    """
    def calculate(self, trajectory):
        angles = np.deg2rad(np.array(trajectory.turning_angles(self,steps_lag=1, normalized=False)))
        
        cos = np.cos(angles)
        sin = np.sin(angles)

        tac = ((cos[1:] - cos[:-1])**2) + ((sin[1:] - sin[:-1])**2)
        tac = np.sum(tac) / trajectory.length

        return [
            tac,
            trajectory.mean_turning_angle(),
            trajectory.correlated_turning_angle(),
            trajectory.directional_persistance(),
        ]
