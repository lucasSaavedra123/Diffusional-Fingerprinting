from .Feature import Feature


class TrajectoryLength(Feature):
    def calculate(self, trajectory):
        return [len(trajectory)]
