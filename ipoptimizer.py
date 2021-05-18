#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file ipoptimizer.py
#  \brief Python script for running ipopt.
#  \author T. Dick

import sys
import numpy as np
import csv
from ipoptconfig import *
from ipoptproblemclass import *
import cyipopt

def Ipoptimizer(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, config):
    """ This function implements a call from FADO
        to the cyipopt python module. """

    # pack everything into a class
    opt_problem = IpoptProblemClass(func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, config)

    # get the boundary arrays
    lb = [config.lb]*config.nparam
    ub = [config.ub]*config.nparam

    # create constraint values
    cl = np.append(np.zeros(config.neqcons), config.lower)
    cu = np.append(np.zeros(config.neqcons), config.upper)

    # call ipopt to create the problem
    nlp = cyipopt.Problem(n=config.nparam, m=(config.neqcons+config.nieqcons),
                        problem_obj=opt_problem,
                        lb=lb, ub=ub, cl=cl, cu=cu)

    # solve the problem
    x, info = nlp.solve(x0)

    return x

#end of Ipoptimizer
