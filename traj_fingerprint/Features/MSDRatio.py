from .Fingerprint_feat_gen import MSDratio

from .Feature import Feature


class MSDRatio(Feature):
    def calculate(self, trajectory):
        return [MSDratio(trajectory.info['msd'])]
