class Feature():
    @property
    def names(self):
        raise NotImplementedError

    def calculate(self, trajectory):
        raise NotImplementedError
