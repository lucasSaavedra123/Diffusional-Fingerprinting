from .Fingerprint_feat_gen import GetMax

from .Feature import Feature


class MaximumDistance(Feature):
    @property
    def names(self):
        return ['Efficiency']

    """
    From (Pinholt, 2019). In (Vogler, 2024)
    is not used the squared value.
    """
    def calculate(self, trajectory):
        return [
            GetMax(trajectory.get_noisy_x(), trajectory.get_noisy_y())
        ]
