from .Fingerprint_feat_gen import Trappedness as rawTrappedness
from .Fingerprint_feat_gen import GetMax

from .Feature import Feature


class Trappedness(Feature):
    @property
    def names(self):
        return ['Trappedness']

    """
    Defined by (Pinholt, 2019) and (Wagner, 2017) and (Kowalek, 2019).
    However, I found other definition by (Kovtun, 2025)
    which do not rely on any model. Check
    which definition to use.
    """
    def calculate(self, trajectory):
        return [rawTrappedness(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            GetMax(trajectory.get_noisy_x(), trajectory.get_noisy_y()),
            trajectory.info['msd']
        )]
