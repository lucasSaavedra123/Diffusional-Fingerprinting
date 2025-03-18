from .Feature import Feature


class Directionality(Feature):
    """
    Obtained from https://doi.org/10.21203/rs.3.rs-3716053/v1
    """
    def calculate(self, trajectory):
        return [
            trajectory.mean_turning_angle(),
            trajectory.correlated_turning_angle(),
            trajectory.directional_persistance(),
        ]
