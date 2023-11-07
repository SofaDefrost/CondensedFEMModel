import numpy as np
USE_QPOASES = False
def init_QP(n_act_constraint, nb_add_constraints = 0):
    """
    Init QP inverse problem for a given configuration

    Parameters
    ----------
    n_act_constraint: int
        Number of actuator constraints
    nb_add_constraints: int
        By default, we only consider constraints on both actuation and actuation displacement.
        Additional constraints must be initialized.

    Outputs
    ----------
    problem: QProblem
        Initialized QProblem
    """
    nb_variables = n_act_constraint

    if USE_QPOASES:
        from qpoases import PyQProblem as QProblem
        from qpoases import PyPrintLevel as PrintLevel
        from qpoases import PyOptions as Options

        nb_constraints = nb_variables + nb_add_constraints
        problem = QProblem(nb_variables, nb_constraints)
        options = Options()
        options.printLevel = PrintLevel.NONE
        problem.setOptions(options)
    # Using proxsuite
    else:
        import proxsuite

        nb_constraints = 2 * nb_variables + nb_add_constraints
        problem = proxsuite.proxqp.sparse.QP(nb_variables, 0, nb_constraints) # number of variables, number of equality constraints, number of inequality constraints
        problem.settings.initial_guess = (proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT)
        problem.settings.max_iter = 2000
    return problem



def solve_QP(problem, H, g, A, lb, ub, lbA, ubA, is_init = True):
    """
    Solve QP inverse for a given configuration
    Ressource: qpoases manual at https://www.coin-or.org/qpOASES/doc/3.2/manual.pdf
               proxsuite manual at https://simple-robotics.github.io/proxsuite/md_doc_2_PROXQP_API_2_ProxQP_api.html
    ----------
    Parameters
    ----------
    problem: QProblem
        Initialized QProblem
    H: numpy array
        Quadratic cost matrix
    g: numpy array
        Linear cost matrix
    A: numpy array
        Constraint matrix
    lb: numpy array
        Lower bound for actuation
    ub: numpy array
        Upper bound for actuation
    lbA: numpy array
        Lower bound for actuation displacement
    ubA: numpy array
        Upper bound for actuation displacement
    is_init: bool
        True if the solver has already been initialized
    ----------
    Outputs
    ----------
    lambda_a: numpy array
        Actuation displacement
    """
    nWSR = 2000 # Number of QP iterations

    n_constraints = len(lb)

    if USE_QPOASES:
        """
        QP problem shape is:
        argmin_x  xT H x + xT g(w0)
            s. t. lbA(w0) ≤ Ax ≤ ubA(w0) ,
                  lb(w0) ≤ x ≤ ub(w0)
        """
        problem.init(H, g, A, lb, ub, lbA, ubA, np.array([nWSR]))

        # Solve
        lambda_a = np.zeros(len(dfree_a))
        problem.getPrimalSolution(lambda_a)

    else: # proxsuite
        """
        QP problem shape is:
        argmin_x  1/2 xT H x + gT(w0) x
            s. t. A(w0)x ≤ b(w0) ,
                  C(w0)x ≤ u(w0)
        """
        # Displacement constraints
        C = A
        l = lbA
        u = ubA

        # Actuation effort constraints
        # Constraint lb(w0) ≤ x ≤ ub(w0) is equivalent to constraints -x ≤ -lb(w0) and x ≤ ub(w0)
        nb_variables = H.shape[0]
        I = np.eye(nb_variables)

        # Final constraint matrices
        C = np.concatenate((C, np.array(I)), axis=0)
        l = np.concatenate([l, lb])
        u = np.concatenate([u, ub])

        if is_init:
            problem.update(H, g, None, None, C, l, u)
        else:
            problem.init(H, g, None, None, C, l, u)

        problem.solve()
        lambda_a = problem.results.x
    return lambda_a
