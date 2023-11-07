#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>

typedef sofa::component::constraint::lagrangian::solver::ConstraintSolverImpl ConstraintSolverImpl;

namespace CondensedFEMModel
{
    class ConstraintSolverSetter : public sofa::core::objectmodel::BaseObject
    {

    public:
        SOFA_CLASS(ConstraintSolverSetter, sofa::core::objectmodel::BaseObject);

    public:
        ConstraintSolverSetter();
        void init() override;

        ConstraintSolverImpl * m_constraintsolver;

        sofa::component::constraint::lagrangian::solver::ConstraintProblem* getConstraintProblem();

        ConstraintSolverImpl* getConstraintSolver()
        {
            return dynamic_cast<ConstraintSolverImpl *>(this->getContext()->get<ConstraintSolverImpl>(sofa::core::objectmodel::BaseContext::SearchDown));
        }

    };

} // namespace CondensedFEMModel
