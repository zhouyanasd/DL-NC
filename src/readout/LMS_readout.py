import numpy as np

from src.readout import Readout

class LMS_readout(Readout ):
    def __int__(self, id):
        Readout.__init__(self, id)

    def LMS(self):


