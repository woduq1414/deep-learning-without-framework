# %%

from framework.functions import *
from framework.gradient import *


# %%

class InitializerType:
    pass


class Std(InitializerType):

    def __init__(self, std):
        self.std = std


class He(InitializerType):
    pass


class Xavier(InitializerType):
    pass




