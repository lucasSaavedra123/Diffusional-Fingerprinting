from .Fingerprint_feat_gen import Efficiency as rawEfficiency

from .Feature import Feature


class Efficiency(Feature):
    def calculate(self, trajectory):
        return [rawEfficiency(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
        )]
