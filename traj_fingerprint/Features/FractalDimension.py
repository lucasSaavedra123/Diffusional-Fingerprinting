from .Fingerprint_feat_gen import FractalDim
from .Fingerprint_feat_gen import GetMax

from .Feature import Feature


class FractalDimension(Feature):
    def calculate(self, trajectory):
        """
        From (Pinholt, 2019). Also is defined
        by (Wagner, 2017) and (Kowalek, 2022), (Kowalek, 2019).
        """
        return [FractalDim(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            GetMax(trajectory.get_noisy_x(), trajectory.get_noisy_y())
        )]
