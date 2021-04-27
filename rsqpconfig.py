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
        self.feasibility_tolerance=1e-7
        self.force_feasibility=False
        self.scale_hessian=False
        self.hybrid_sobolev=False
        self.bfgs=None

    # end
