#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file analyzehessian.py
#  \brief Python script for finite difference hessian analysis.
#  \author T. Dick

import sys
import numpy as np
import csv

def analyzehessian(x0, func, f_eqcons, fprime, fprime_eqcons, fdstep):
    """ This is a implementation
        to calcuate Hessians with finite differences."""

    p = x0

    #compute the base vectors
    D_F_base = fprime(p)
    D_E_base = fprime_eqcons(p)

    D_F_data = np.zeros([len(p),len(p)])
    D_E_data = np.zeros([len(p),len(p)])

    for dir in range(len(p)):

        p = np.zeros(len(p))
        p[dir] = fdstep

        D_F = fprime(p)
        D_E = fprime_eqcons(p)

        D_F_data[:,dir] = D_F
        D_E_data[:,dir] = D_E

        np.savetxt("D_F_data.csv",D_F_data,delimiter=",")
        np.savetxt("D_E_data.csv",D_E_data,delimiter=",")

    H_F = computehess(D_F_data, D_F_base, fdstep)
    H_E = computehess(D_E_data, D_E_base, fdstep)

    np.savetxt("H_F.csv",H_F,delimiter=",")
    np.savetxt("H_E.csv",H_E,delimiter=",")

    print("HAHA")

    # end of for

# end of def


def computehess(data,base,step):

    hess = np.zeros([len(base),len(base)])

    for i in range(len(base)):
        for j in range(len(base)):
            hess[i,j] = (data[i,j]-base[i])/(2*step) + (data[j,i]-base[j])/(2*step)

    return 0.5*(hess+np.transpose(hess))

#end of def
