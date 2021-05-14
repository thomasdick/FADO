#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file ipoptimizer.py
#  \brief Python script for running ipopt.
#  \author T. Dick

import sys
import numpy as np
import csv
from rsqpconfig import *

def Ipoptimizer(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, iter, acc, lsmode, config, xb=None, driver=None):
    """ This function implements a call from FADO
        to the cyipopt python module. """

    # pack everything into a class
    opt_problem = IpoptProblemClass( )

    # call ipopt
    nlp = ipopt.problem(n=len(x0), m=len(cl),
                        problem_obj=opt_problem,
                        lb=lb, ub=ub, cl=cl, cu=cu)

    return 0

#end of Ipoptimizer
