from .Fingerprint_feat_gen import Efficiency as rawEfficiency

from .Feature import Feature


class Efficiency(Feature):
    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017) and (Kowalek, 2022), (Kowalek, 2019).
    """
    def calculate(self, trajectory):
        return [rawEfficiency(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
        )]
