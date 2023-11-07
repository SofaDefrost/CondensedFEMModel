#pragma once

#include "ConstraintSolverSetter.h"

namespace CondensedFEMModel
{

    ConstraintSolverSetter::ConstraintSolverSetter()
    {}

    void ConstraintSolverSetter::init()
    {
        m_constraintsolver = getConstraintSolver();
        if (m_constraintsolver == NULL){
              msg_error() << "Error cannot find the constraint solver.";
        }
    }

    sofa::component::constraint::lagrangian::solver::ConstraintProblem* ConstraintSolverSetter::getConstraintProblem()
    {
        return m_constraintsolver->getConstraintProblem();
    }

} // Sofa
