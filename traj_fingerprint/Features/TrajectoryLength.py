from .Feature import Feature


class TrajectoryLength(Feature):
    @property
    def names(self):
        return [r'$L$']

    def calculate(self, trajectory):
        return [trajectory.length]
