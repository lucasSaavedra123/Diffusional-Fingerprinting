from .Fingerprint_feat_gen import Gaussianity as rawGaussianity

from .Feature import Feature


class Gaussianity(Feature):
    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017).
    """
    def calculate(self, trajectory):
        return [rawGaussianity(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            trajectory.info['msd']
        )]
