from .Fingerprint_feat_gen import Trappedness as rawTrappedness
from .Fingerprint_feat_gen import GetMax

from .Feature import Feature


class Trappedness(Feature):
    def calculate(self, trajectory):
        return [rawTrappedness(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            GetMax(trajectory.get_noisy_x(), trajectory.get_noisy_y()),
            trajectory.info['msd']
        )]
