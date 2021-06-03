#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

import sys
import numpy as np
from scipy import optimize

class RSQPconfig:
    """ This calls is for storing options and handing them to the reduced SQP optimizer.
        Config options are stored as members of this class.
    """

    def __init__(self):

        # scaling and feasibility options
        self.feasibility_tolerance=1e-7
        self.force_feasibility=False
        self.scale_hessian=False

        # identity part of the Hessian operator
        self.hybrid_sobolev=False
        self.epsilon3=1.0

        # BFGS options
        self.bfgs=None
        self.bfgscons=None

        # merit function options
        self.meritfunction=False
        self.mfchoice=1
        self.nu=17.85
        self.delta=0.0025
        self.rho=np.array([0])

        # linesearch options
        self.steps=None

    # end init

# end RSQPconfig
