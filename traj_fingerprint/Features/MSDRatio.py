from .Fingerprint_feat_gen import MSDratio

from .Feature import Feature


class MSDRatio(Feature):
    @property
    def names(self):
        return ['MSDRatio']

    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017) and (Kowalek, 2022), (Kowalek, 2019).
    """
    def calculate(self, trajectory):
        return [MSDratio(trajectory.info['msd'])]
