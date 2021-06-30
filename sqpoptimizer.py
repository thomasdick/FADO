#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file sqpoptimizer.py
#  \brief Python script for performing the SQP optimization with a semi handwritten optimizer.
#  \author T. Dick

import sys
import numpy as np
import cvxopt
import csv
from rsqpconfig import *

def SQPconstrained(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, iter, acc, lsmode, config, xb=None, driver=None):
    """ This is a implementation of a SQP optimizer
        It is written for smoothed derivatives
        Accepts:
            - equality and inequality constraints
            - approximated hessians
            - additional parameters
        can be applied to analytical functions for test purposes"""

    sys.stdout.write('Using the hand implemented version. Setting up the environment...' + '\n')

    # set the inout parameters
    p = x0
    err = 2*acc+1
    step = 1

    # extract design parameter bounds
    if xb is None:
        xb = [[-1e-1]*len(p), [1e-1]*len(p)]
    else:
        #unzip the list
        xb = list(zip(*xb))

    # prepare output, using 'with' command to automatically close the file in case of exceptions
    with open("optimizer_history.csv", "w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        header = ['iter', 'objective function', 'equal constraint', 'inequal constraint', 'design', 'norm(fullstep)', 'norm(delta_p)', 'lm_eqcons', 'lm_ieqcons', 'Lagrangian','gradLagrangian','nEval']
        csv_writer.writerow(header)

        # main optimizer loop
        while (err > acc and step <= iter):

            sys.stdout.write('Optimizer iteration: ' + str(step) + '\n')

            # evaluate the functions
            F = func(p)
            E = f_eqcons(p)
            C = f_ieqcons(p)
            D_F = fprime(p)
            D_E = fprime_eqcons(p)
            D_C = fprime_ieqcons(p)

            # Hessian computation
            H_F = fdotdot(p)
            H_F = 0.5*(H_F+np.transpose(H_F))
            if config.scale_hessian:
                H_F = H_F/np.linalg.norm(H_F,2)

            if config.hybrid_sobolev:
                if config.bfgs == None:
                    if (np.size(config.epsilon3) > 1):
                        H_F = H_F + config.epsilon3[step]*np.identity(len(p))
                    else:
                        H_F = H_F + config.epsilon3*np.identity(len(p))
                else:
                    # testwise BFGS initialization???
                    if step == 1:
                        config.bfgs.B = H_F + config.epsilon3*np.identity(len(p))
                        H_F = config.bfgs.get_matrix()
                    if step > 1:
                        config.bfgs.update(delta_p, ((D_F-oldDF).flatten()+lm_eqcons *(D_E-oldDE).flatten()))
                        H_F = config.bfgs.get_matrix()
                    oldDF = D_F
                    oldDE = D_E

            # assemble equality constraints
            if np.size(E) > 0:
                A = cvxopt.matrix(D_E)
                b = cvxopt.matrix(-E)

            # expand inequality constraints by bounds
            Id = np.identity(len(p))
            if np.size(C) > 0:
                G = cvxopt.matrix(np.block([[-D_C], [-Id], [Id]]))
                h = cvxopt.matrix(np.append(C,np.append( -np.array(xb[0]), np.array(xb[1]))))
            else:
                G = cvxopt.matrix(np.block([[-Id], [Id]]))
                h = cvxopt.matrix(np.append( -np.array(xb[0]), np.array(xb[1])))

            # pack objective function
            P = cvxopt.matrix(H_F)
            q = cvxopt.matrix(D_F)

            # solve the interior quadratic problem
            if np.size(E) > 0:
                sol = cvxopt.solvers.qp(P, q, G, h, A, b, feastol=config.feasibility_tolerance)
            else:
                sol = cvxopt.solvers.qp(P, q, G, h, feastol=config.feasibility_tolerance)

            # extract values from the quadratic solver
            # maybe we should use negative sign because how the QP is set up
            delta_p = np.array([i for i in sol['x']])
            lm_eqcons = np.array([i for i in sol['y']])
            lm_ieqcons = np.zeros(np.size(C))
            for i in range(np.size(C)):
                lm_ieqcons[i] = sol['z'][i]
            sol_length=np.linalg.norm(delta_p,2)

            # usually we would use the Lagrangian from last iteration, but in 1 step it is not available.
            if (step==1):
                Lagrang = Lagrangian(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons)

            # line search
            delta_p = linesearch(p, delta_p, F, Lagrang, D_F, D_E, D_C, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, acc, lsmode, step, config)

            # update the Lagrangian for linesearch in the next iteration
            Lagrang = Lagrangian(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons)
            gradL = D_F + lm_eqcons @ D_E + lm_ieqcons @ D_C

            # update the design
            p = p + delta_p
            err = np.linalg.norm(delta_p, 2)

            sys.stdout.write('New design: ' + str(p) + '\n')
            if(err<=acc):
                sys.stdout.write('Reached convergence criteria. \n')

            # write to the history file
            if driver!=None:
                nEval=driver._funEval
            else:
                nEval=0

            line = [step, F, E, C, p, sol_length, err, lm_eqcons, lm_ieqcons, Lagrang, gradL, nEval]
            csv_writer.writerow(line)
            outfile.flush()

            # additional step in direction of feasibility
            if config.force_feasibility:
                sys.stdout.write('Additional feasibility step:')

                # evaluate the constraints in the new point
                E = f_eqcons(p)
                C = f_ieqcons(p)
                D_E = fprime_eqcons(p)
                D_C = fprime_ieqcons(p)

                # assemble equality constraints
                if np.size(E) > 0:
                    A = cvxopt.matrix(D_E)
                    b = cvxopt.matrix(-E)

                # expand inequality constraints by bounds
                Id = np.identity(len(p))
                if np.size(C) > 0:
                    G = cvxopt.matrix(np.block([[-D_C], [-Id], [Id]]))
                    h = cvxopt.matrix(np.append(C,np.append( -np.array(xb[0]), np.array(xb[1]))))
                else:
                    G = cvxopt.matrix(np.block([[-Id], [Id]]))
                    h = cvxopt.matrix(np.append( -np.array(xb[0]), np.array(xb[1])))

                # minimize distance as objective function
                P = cvxopt.matrix(Id)
                q = cvxopt.matrix(np.zeros(len(p)))

                # solve the interior quadratic problem
                if np.size(E) > 0:
                    sol = cvxopt.solvers.qp(P, q, G, h, A, b, feastol=config.feasibility_tolerance)
                else:
                    sol = cvxopt.solvers.qp(P, q, G, h, feastol=config.feasibility_tolerance)

                delta_p = np.array([i for i in sol['x']])
                norm = np.linalg.norm(delta_p, 2)
                if (norm>=lsmode):
                    delta_p = (lsmode/norm) * delta_p
                p = p + delta_p

            # increase counter at the end of the loop
            step += 1

    # end output automatically

    return 0

# end SQPconstrained


def SQPequalconstrained(x0, func, f_eqcons, fprime, fprime_eqcons, fdotdot, iter, acc, lsmode, config, xb=None, driver=None):
    """ This is a implementation of a SQP optimizer
        It is written for smoothed derivatives
        Accepts:
            - only equality constraints
            - approximated hessians
            - additional parameters
        can be applied to analytical functions for test purposes"""

    sys.stdout.write('Using the simplified hand implemented version for equality constraints. Setting up the environment...' + '\n')

    # set the inout parameters
    p = x0
    err = 2*acc+1
    step = 1

    # prepare output, using 'with' command to automatically close the file in case of exceptions
    with open("optimizer_history.csv", "w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        header = ['iter', 'objective function', 'equal constraint', 'design', 'norm(fullstep)', 'norm(delta_p)', 'lm_eqcons', 'Lagrangian', 'gradLagrangian', 'nEval']
        csv_writer.writerow(header)

        # main optimizer loop
        while (err > acc and step <= iter):

            sys.stdout.write('Optimizer iteration: ' + str(step) + '\n')

            # reevaluate the functions
            F = func(p)
            E = f_eqcons(p)
            D_F = fprime(p)
            D_E = fprime_eqcons(p)
            H_F = fdotdot(p)

            # assemble linear equation system
            rhs = np.append([-D_F], [-E])
            mat = np.block([[H_F, D_E.T], [D_E, np.zeros((np.size(E),np.size(E)))]])

            # solve the LES
            sol = np.linalg.solve(mat, rhs)

            # FIND way to incorporate boundary
            # G = cvxopt.matrix(np.block([[-Id], [Id]]))
            # h = cvxopt.matrix(np.append([-xb[0]]*len(p), [xb[1]]*len(p)))

            # get the solution
            delta_p = sol[0:len(p)]
            lm_eqcons = sol[-np.size(E):]
            sol_length = np.linalg.norm(delty_p,2)

            # usually we would use the Lagrangian from last iteration, but in 1 step it is not available.
            if (step==1):
                Lagrangian = F + np.inner(E,lm_eqcons)

            # line search
            delta_p = linesearch(p, delta_p, F, Lagrangian, D_F, D_E, D_C, func, f_eqcons, empty_func, lm_eqcons, 0, acc, lsmode, step, config)

            # update the Lagrangian for linesearch in the next iteration
            Lagrangian = F + np.inner(E,lm_eqcons)
            gradL = D_Lagrangian(p, D_F, D_E, D_C, f_ieqcons, lm_eqcons, lm_ieqcons)

            # update the design
            p = p + delta_p
            err = np.linalg.norm(delta_p, 2)

            sys.stdout.write('New design: ' + str(p) + '\n')
            if(err<=acc):
                sys.stdout.write('Reached convergence criteria. \n')

            # write to the history file
            if driver!=None:
                nEval=driver._funEval
            else:
                nEval=0

            line = [step, F, E, p, sol_length, err, lm_eqcons, Lagrangian, gradL, nEval]
            csv_writer.writerow(line)
            outfile.flush()

            # increase counter at the end of the loop
            step += 1

    # end output automatically

    return 0

# end SQPequalconstrained


def linesearch(p, delta_p, F, Lprev, D_F, D_E, D_C, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, acc, lsmode, step, config):

    if (np.size(config.steps) != 0):
        mode = config.steps[step]
    else:
        mode = lsmode

    #merit function linesearch
    if (config.meritfunction):

        # calculate merit function
        # option 1: L1-norm
        # option 2: SLSQP type merit function
        if (config.mfchoice == 1):

            # guess the initial step
            alpha = lsmode
            p_new = p + alpha*delta_p

            M = merit(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config.nu)
            # direct D_M = D_F makes D_M a reference to D_F, changes then always affect both!
            D_M = np.zeros(np.size(D_F))
            D_M += D_F
            for i in range(np.size(f_eqcons(p))):
                if f_eqcons(p)[i]>=0.0:
                    D_M = D_M + D_E[i] / config.nu
                else:
                    D_M = D_M - D_E[i] / config.nu
            M_new = merit(p_new, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config.nu)

            #reduce step and repeat
            while M_new > M + 0.5*alpha*np.inner(delta_p,D_M):
                alpha = alpha/5
                p_new = p + alpha*delta_p
                M_new = merit(p_new, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config.nu)
                # avoid the step getting too small
                if alpha<1e-6:
                    break

            #adapt nu
            if (1/config.nu) > np.linalg.norm(lm_eqcons, np.inf) + config.delta:
                config.nu = 1/(np.linalg.norm(lm_eqcons, np.inf) + 2*config.delta)
                sys.stdout.write("new nu: " + str(config.nu) + "\n")

            return alpha*delta_p

        elif (config.mfchoice == 2):

            sys.stdout.write("Using SLSQP style merit function. \n")

            # evaluate the merit function
            feq = f_eqcons(p)
            fieq = f_ieqcons(p)
            M = merit2(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config)
            D_M = d_merit2(p, D_F, D_E, D_C, feq, fieq, config)

            # adapt rho
            for i in range(np.size(f_eqcons(p))):
                config.rho[i] = np.maximum( 0.5*(config.rho[i]+np.absolute(lm_eqcons[i])), np.absolute(lm_eqcons[i]))
            for j in range(np.size(feq)):
                config.rho[j+np.size(f_eqcons(p))] = np.maximum( 0.5*(config.rho[j+np.size(f_eqcons(p))]+np.absolute(lm_ieqcons[j])), np.absolute(lm_ieqcons[j]))

            # Newton step
            alpha = -M / np.inner(D_M,delta_p/np.linalg.norm(delta_p))
            p_new = p + alpha*delta_p
            M_new = merit2(p_new, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config)
            count=1
            while M_new > 1e-6:
                alpha = -M_new / np.inner(D_M,delta_p/np.linalg.norm(delta_p))
                p_new = p_new + alpha*delta_p
                M_new = merit2(p_new, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config)
                # avoid too many Newton steps
                if count>=5:
                    break
                count=count+1

            return alpha*delta_p

    # end of merit function linesearch

    # use a maximum step length
    if (mode >= 0.0):
        norm = np.linalg.norm(delta_p, 2)
        if (norm>=mode):
            delta_p = (mode/norm) * delta_p

    # use an increased step
    elif (mode <= -3.0):
        factor = -(mode+3.0)
        delta_p = factor * delta_p

    # backtracking based on objective function
    elif (mode == -1.0):
        criteria = True
        while (criteria):
            p_temp = p + delta_p
            sys.stdout.write("descend step: " + str(delta_p) + "\n")
            F_new = func(p_temp)
            sys.stdout.write("old objective: " + str(F) + " , new objective: " + str(F_new) + "\n")
            # test for optimization progress
            if (F_new > F):
                sys.stdout.write("not a reduction in objective function. \n")
                delta_p = 0.1*delta_p
            else:
                sys.stdout.write("descend step accepted. \n")
                criteria = False

            #avoid step getting to small
            if (np.linalg.norm(delta_p, 2) < acc):
                sys.stdout.write("can't find a good step. \n")
                criteria = False

    # backtracking based on the Laplacian
    elif (mode == -2.0):

        alpha = config.lbtalpha
        orig_norm = np.linalg.norm(delta_p)

        # get Lagrangian and sensitivity with current multipliers
        L = Lagrangian(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons)
        gradL = D_Lagrangian(p, D_F, D_E, D_C, f_ieqcons, lm_eqcons, lm_ieqcons)

        criteria = True
        while (criteria):

            # compute Lagrangian at new position
            p_temp = p + (alpha/orig_norm)*delta_p
            L_new = Lagrangian(p_temp, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons)

            sys.stdout.write("old Lagrangian: " + str(L) + " , new Lagrangian: " + str(L_new) + "\n")
            # test for optimization progress
            if (L_new > L + 1e-4*(alpha/orig_norm)*np.inner(delta_p, gradL)):
                sys.stdout.write("not a reduction in Lagrangian function. \n")
                alpha = alpha / config.lbtdelta
            else:
                sys.stdout.write("descend step accepted. \n")
                criteria = False

            #avoid step getting to small
            if (alpha <= 1e-4):
                sys.stdout.write("can't find a good step. \n")
                criteria = False
                alpha=1e-4

        delta_p = (alpha/orig_norm)*delta_p

    # if unknown mode choosen, leave direction unaffected.
    else:
        sys.stdout.write("Unknown line search mode. leave search direction unaffected. \n")
        sys.stdout.write("descend step: " + str(delta_p) + "\n")

    return delta_p

# end of linesearch


# we need a unit matrix to have a dummy for optimization tests.
def unit_hessian(x):
    return np.identity(np.size(x))
# end of unit_hessian


# we need an empty function call to have a dummy for optimization tests.
def empty_func(x):
    return 0
# end of empty_func


def Lagrangian(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons):
    Lvalue = func(p) + np.inner(f_eqcons(p), lm_eqcons)
    fieq = f_ieqcons(p)
    for j in range(np.size(fieq)):
        Lvalue += lm_ieqcons[j]*np.minimum( 0.0, fieq[j])
    return Lvalue
#end Lagrangian


def D_Lagrangian(p, D_F, D_E, D_C, f_ieqcons, lm_eqcons, lm_ieqcons):
    gradLvalue = D_F + lm_eqcons @ D_E
    fieq = f_ieqcons(p)
    for j in range(np.size(fieq)):
        if (fieq[j] <= 0.0):
            gradLvalue += lm_ieqcons[j]*D_C[j,:]
    return gradLvalue
#end D_Lagrangian


#implementation of the L1-norm merit function
def merit(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, nu):
    M = func(p) + np.linalg.norm(f_eqcons(p),1) /nu
    return M
#end of merit


# implementation of the SLSQP style merit function
def merit2(p, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, config):
    M=func(p)
    feq = f_eqcons(p)
    fieq = f_ieqcons(p)
    for i in range(np.size(feq)):
        M += config.rho[i]*np.absolute(feq[i])
    for j in range(np.size(fieq)):
        M += config.rho[j+np.size(feq)]*np.absolute( np.minimum( 0.0, fieq[j]) )
    return M
#end of merit2


# evaluate the derivative of the SLSQP style merit function
def d_merit2(p, D_F, D_E, D_C, feq, fieq, config):
    # direct D_M = D_F makes D_M a reference to D_F, changes then always affect both!
    D_M = np.zeros(np.size(D_F))
    D_M += D_F
    for i in range(np.size(feq)):
        if feq[i] >= 0:
            D_M += config.rho[i]*D_E[i,:]
        else:
            D_M -= config.rho[i]*D_E[i,:]
    for j in range(np.size(fieq)):
        if (fieq[j] <= 0.0):
            D_M -= config.rho[j+np.size(feq)]*D_C[j,:]
    return D_M
#end of d_merit2
