#include <pybind11/pybind11.h>
#include "Binding_CondensedFEMModel.h"


namespace py { using namespace pybind11; }

namespace sofapython3
{

PYBIND11_MODULE(CondensedFEMModel, m)
{
    moduleAddCondensedFEMModel(m);
}

} // namespace sofapython3
