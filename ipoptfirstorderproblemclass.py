#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file ipoptimizer.py
#  \brief Python script for running ipopt.
#  \author T. Dick

import sys
import numpy as np
import csv
from ipoptconfig import *

class IpoptFirstOrderProblemClass:
    """ Implement the interface of a cyipopt problem class.
        Pack the FADO calls together for Ipopt. """

    def __init__(self, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons):
        self._func = func
        self._f_eqcons = f_eqcons
        self._f_ieqcons = f_ieqcons
        self._fprime = fprime
        self._fprime_eqcons = fprime_eqcons
        self._fprime_ieqcons = fprime_ieqcons
    #end init

    def objective(self, x):
        return self._func(x)

    def gradient(self, x):
        return self._fprime(x)

    def constraints(self, x):
        return np.append(self._f_eqcons(x), self._f_ieqcons(x))

    def jacobian(self, x):
        return np.append(self._fprime_eqcons(x), self._fprime_ieqcons(x))

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm,
                     regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

# end IpoptProblemClass
