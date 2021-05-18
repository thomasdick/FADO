#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

import sys
import numpy as np

class Ipoptconfig:
    """ This class is a container for storing options for IpOpt
    """

    def __init__(self):
        self.nparam=1
        self.neqcons=0
        self.nieqcons=0
        self.lb=-1.0
        self.ub=1.0
        self.lower=None
        self.upper=None

    # end init

# end RSQPconfig
