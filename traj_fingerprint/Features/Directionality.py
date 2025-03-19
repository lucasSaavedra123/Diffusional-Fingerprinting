import numpy as np

from .Feature import Feature


class Directionality(Feature):
    @property
    def names(self):
        return ['tac', 'sinuosity', 'meanDP', 'corrDP', 'AvgSignDp']

    """
    Obtained from https://doi.org/10.21203/rs.3.rs-3716053/v1
    and https://doi.org/10.1039/D3NA00188A
    """
    def calculate(self, trajectory):
        angles = np.deg2rad(np.array(trajectory.turning_angles(steps_lag=1, normalized=False)))

        cos = np.cos(angles)
        sin = np.sin(angles)

        tac = ((cos[1:] - cos[:-1])**2) + ((sin[1:] - sin[:-1])**2)
        tac = np.sum(tac) / trajectory.length

        ratio_a = 1-(np.mean(cos)**2)-(np.mean(sin)**2)
        ratio_b = ((1-np.mean(cos))**2)+(np.mean(sin)**2)
        b = np.std(trajectory.displacements())/np.mean(trajectory.displacements())
        ratio_d = (ratio_a/ratio_b) + (b**2)

        sinuosity = 2*((np.mean(trajectory.displacements()) * ratio_d)**(-0.5))

        return [
            tac,
            sinuosity,
            trajectory.mean_turning_angle(),
            trajectory.correlated_turning_angle(),
            trajectory.directional_persistance(),
        ]
