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
from ipoptfirstorderproblemclass import *
import cyipopt

def Ipoptimizer(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, config):
    """ This function implements a call from FADO
        to the cyipopt python module. """

    # pack everything into a class
    opt_problem = IpoptProblemClass(func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, config)

    if fdotdot == None:
        opt_problem = IpoptFirstOrderProblemClass(func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons)

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

    nlp.addOption('nlp_scaling_method', 'none')

    # solve the problem
    x, info = nlp.solve(x0)

    return x
#end of Ipoptimizer

# we need a unit matrix to have a dummy for optimization tests.
def unit_hessian(x):
    return np.identity(np.size(x))
# end of unit_hessian
