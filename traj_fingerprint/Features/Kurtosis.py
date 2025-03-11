from .Fingerprint_feat_gen import Kurtosis as rawKurtosis

from .Feature import Feature


class Kurtosis(Feature):
    def calculate(self, trajectory):
        return [rawKurtosis(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
        )]
