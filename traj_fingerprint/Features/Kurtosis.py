from .Fingerprint_feat_gen import Kurtosis as rawKurtosis

from .Feature import Feature


class Kurtosis(Feature):
    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017) and (Kowalek, 2022).
    """
    def calculate(self, trajectory):
        return [rawKurtosis(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
        )]
