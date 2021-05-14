#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file ipoptimizer.py
#  \brief Python script for running ipopt.
#  \author T. Dick

import sys
import numpy as np
import csv
from rsqpconfig import *


class IpoptProblemClass:
    """ Implement the interface of a cyipopt problem class.
        Pack the FADO calls together for Ipopt. """

    def __init__(self, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot):
        self._func = func
        self._f_eqcons = f_eqcons
        self._f_ieqcons = f_ieqcons
        self._fprime = fprime
        self._fprime_eqcons = fprime_eqcons
        self._fprime_ieqcons = fprime_ieqcons
        self._fdotdot = fdotdot
    #end init

    def objective(self, x):
        return self._func(x)

    def gradient(self, x):
        return self._f_prime(x)

    def constraints(self, x):
        return np.append(self._f_eqcons(x), self._f_ieqcons(x))

    def jacobian(self, x):
        return np.append(self._fprime_eqcons(x), self._fprime_ieqcons(x))

    def hessianstructure(self):
            #
            # The structure of the Hessian
            # Note:
            # The default hessian structure is of a lower triangular matrix. Therefore
            # this function is redundant. I include it as an example for structure
            # callback.
            #

            return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
            #
            # The callback for calculating the Hessian
            #
            H = obj_factor*np.array((
                    (2*x[3], 0, 0, 0),
                    (x[3],   0, 0, 0),
                    (x[3],   0, 0, 0),
                    (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

            H += lagrange[0]*np.array((
                    (0, 0, 0, 0),
                    (x[2]*x[3], 0, 0, 0),
                    (x[1]*x[3], x[0]*x[3], 0, 0),
                    (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

            H += lagrange[1]*2*np.eye(4)

            row, col = self.hessianstructure()

            return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm,
                     regularization_size,
                     alpha_du, alpha_pr, ls_trials^):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

# end IpoptProblemClass
