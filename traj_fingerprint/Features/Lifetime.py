from .Fingerprint_feat_gen import Lifetime as rawLifetime
from .Fingerprint_feat_gen import GetStates

from .Feature import Feature


class Lifetime(Feature):
    def __init__(self, model):
        self.__model = model

    @property
    def names(self):
        return [r'$LifeTime$']

    def calculate(self, trajectory):
        states, _ = GetStates(trajectory.displacements(), self.__model)
        return [rawLifetime(states)]