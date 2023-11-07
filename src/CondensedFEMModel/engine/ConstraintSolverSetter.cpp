#include "ConstraintSolverSetter.inl"
#include <sofa/core/ObjectFactory.h>

namespace CondensedFEMModel
{

    SOFA_DECL_CLASS(ConstraintSolverSetter)

    int ConstraintSolverSetterClass = sofa::core::RegisterObject("Set mechanical matrices (W, dfree, lambda) in the  constraint solver.")
                                .add<ConstraintSolverSetter>();

} // namespace CondensedFEMModel
