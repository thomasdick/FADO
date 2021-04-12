#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file sqpoptimizer.py
#  \brief Python script for performing the SQP optimization with a semi handwritten optimizer.
#  \author T. Dick

import sys
import numpy as np
import cvxopt
import csv


def SQPconstrained(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, iter, acc, lsmode, xb=None, driver=None, feasibility_tolerance=1e-7, force_feasibility=False, scale_hessian=False, hybrid_sobolev=False):
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
        header = ['iter', 'objective function', 'equal constraint', 'inequal constraint', 'design', 'norm(gradient)', 'norm(delta_p)', 'lm_eqcons', 'lm_ieqcons', 'Lagrangian','gradLagrangian','nEval']
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
            H_F = fdotdot(p)

            # force Hessian to be symmetric
            H_F = 0.5*(H_F+np.transpose(H_F))

            # adapt the scaling of H_F
            if scale_hessian:
                H_F = H_F/np.linalg.norm(H_F,2)

            # experimental change
            if hybrid_sobolev:
                H_F = H_F + np.identity(len(p))

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
                sol = cvxopt.solvers.qp(P, q, G, h, A, b, feastol=feasibility_tolerance)
            else:
                sol = cvxopt.solvers.qp(P, q, G, h, feastol=feasibility_tolerance)

            # extract values from the quadratic solver
            # maybe we should use negative sign because how the QP is set up
            delta_p = np.array([i for i in sol['x']])
            lm_eqcons = np.array([i for i in sol['y']])
            lm_ieqcons = np.zeros(np.size(C))
            for i in range(np.size(C)):
                lm_ieqcons[i] = sol['z'][i]

            # usually we would use the Lagrangian from last iteration, but in 1 step it is not available.
            if (step==1):
                Lagrangian = F + np.inner(E,lm_eqcons) + np.inner(C,lm_ieqcons)

            # line search
            delta_p = linesearch(p, delta_p, F, Lagrangian, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, acc, lsmode)

            # update the Lagrangian for linesearch in the next iteration
            Lagrangian = F + np.inner(E,lm_eqcons) + np.inner(C,lm_ieqcons)
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

            line = [step, F, E, C, p, np.linalg.norm(D_F, 2), err, lm_eqcons, lm_ieqcons, Lagrangian, gradL, nEval]
            csv_writer.writerow(line)
            outfile.flush()

            # additional step in direction of feasibility
            if force_feasibility:
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
                    sol = cvxopt.solvers.qp(P, q, G, h, A, b, feastol=feasibility_tolerance)
                else:
                    sol = cvxopt.solvers.qp(P, q, G, h, feastol=feasibility_tolerance)

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


def SQPequalconstrained(x0, func, f_eqcons, fprime, fprime_eqcons, fdotdot, iter, acc, lsmode, xb=None, driver=None):
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
        header = ['iter', 'objective function', 'equal constraint', 'design', 'norm(gradient)', 'norm(delta_p)', 'lm_eqcons', 'Lagrangian', 'gradLagrangian', 'nEval']
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

            # usually we would use the Lagrangian from last iteration, but in 1 step it is not available.
            if (step==1):
                Lagrangian = F + np.inner(E,lm_eqcons)

            # line search
            delta_p = linesearch(p, delta_p, F, Lagrangian, func, f_eqcons, empty_func, lm_eqcons, 0, acc, lsmode)

            # update the Lagrangian for linesearch in the next iteration
            Lagrangian = F + np.inner(E,lm_eqcons)
            gradL = D_F + lm_eqcons @ D_E

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

            line = [step, F, E, p, np.linalg.norm(D_F, 2), err, lm_eqcons, Lagrangian, gradL, nEval]
            csv_writer.writerow(line)
            outfile.flush()

            # increase counter at the end of the loop
            step += 1

    # end output automatically

    return 0

# end SQPequalconstrained


def linesearch(p, delta_p, F, L, func, f_eqcons, f_ieqcons, lm_eqcons, lm_ieqcons, acc, lsmode):

    mode = lsmode

    # use a maximum step length
    if (mode >= 0.0):
        norm = np.linalg.norm(delta_p, 2)
        if (norm>=mode):
            delta_p = (mode/norm) * delta_p

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
        criteria = True
        while (criteria):
            p_temp = p + delta_p
            F_new = func(p_temp)
            E_new = f_eqcons(p_temp)
            C_new = f_ieqcons(p_temp)
            L_new = F_new + np.dot(lm_eqcons, E_new) + np.dot(lm_ieqcons, C_new)
            sys.stdout.write("old Lagrangian: " + str(L) + " , new Lagrangian: " + str(L_new) + "\n")
            # test for optimization progress
            if (L_new > L):
                sys.stdout.write("not a reduction in Lagrangian function. \n")
                delta_p = 0.1*delta_p
            else:
                sys.stdout.write("descend step accepted. \n")
                criteria = False

            #avoid step getting to small
            if (np.linalg.norm(delta_p, 2) < acc):
                sys.stdout.write("can't find a good step. \n")
                criteria = False

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
