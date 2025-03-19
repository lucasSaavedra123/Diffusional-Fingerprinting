from .Fingerprint_feat_gen import MSDratio

from .Feature import Feature


class MSDRatio(Feature):
    """
    From (Pinholt, 2019). Also is defined
    by (Wagner, 2017).
    """
    def calculate(self, trajectory):
        return [MSDratio(trajectory.info['msd'])]
