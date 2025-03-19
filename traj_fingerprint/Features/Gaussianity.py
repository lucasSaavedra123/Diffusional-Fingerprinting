from .Fingerprint_feat_gen import Gaussianity as rawGaussianity

from .Feature import Feature


class Gaussianity(Feature):
    @property
    def names(self):
        return ['Gaussianity']

    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017) and (Kowalek, 2022), (Kowalek, 2019).
    Check last paper if there are diffrerences
    with implemented method.
    """
    def calculate(self, trajectory):
        return [rawGaussianity(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            trajectory.info['msd']
        )]
