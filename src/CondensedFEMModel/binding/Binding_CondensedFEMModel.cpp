#include "Binding_CondensedFEMModel.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <SofaPython3/Sofa/Core/Binding_Base.h>
#include <SofaPython3/PythonFactory.h>
#include <CondensedFEMModel/engine/ConstraintSolverSetter.h>

namespace py { using namespace pybind11; }

namespace sofapython3 {

typedef CondensedFEMModel::ConstraintSolverSetter ConstraintSolverSetter;
using EigenDenseMatrix = Eigen::Matrix<SReal, Eigen::Dynamic, Eigen::Dynamic>;
using EigenMatrixMap = Eigen::Map<EigenDenseMatrix>;

void moduleAddCondensedFEMModel(py::module &m)
{
    py::class_<ConstraintSolverSetter,
               sofa::core::objectmodel::BaseObject,
               sofapython3::py_shared_ptr<ConstraintSolverSetter> > c(m, "ConstraintSolverSetter");

    c.def("setW", [](ConstraintSolverSetter& self, py::array_t<SReal> _W)
    {
        auto& W_matrix = self.getConstraintProblem()->W;

        auto r = _W.unchecked<2>(); // _W must have ndim = 2; can be non-writeable
        if (r.shape(0) != W_matrix.rows())
            throw py::type_error("Invalid row dimension");
        if (r.shape(1) != W_matrix.cols())
            throw py::type_error("Invalid col dimension");

        std::memcpy(W_matrix.ptr(), _W.data(), W_matrix.rows() * W_matrix.cols() * sizeof(SReal));
    });

    c.def("set_lambda_force", [](ConstraintSolverSetter& self, py::array_t<SReal> _lambda)
    {
        assert(self.getConstraintProblem());
        auto& lambda = self.getConstraintProblem()->f;

        auto r = _lambda.unchecked<1>();
        if (r.shape(0) != lambda.size())
            throw py::type_error("Invalid dimension");

        std::memcpy(lambda.ptr(), _lambda.data(), _lambda.size() * sizeof(SReal));

    });

    c.def("set_dfree", [](ConstraintSolverSetter& self, py::array_t<SReal> _dfree)
    {
        assert(self.getConstraintProblem());
        auto& dFree = self.getConstraintProblem()->dFree;

        auto r = _dfree.unchecked<1>();
        if (r.shape(0) != dFree.size())
            throw py::type_error("Invalid dimension");

        std::memcpy(dFree.ptr(), _dfree.data(), _dfree.size() * sizeof(SReal));

    });

    /// register the binding in the downcasting subsystem
    sofapython3::PythonFactory::registerType<ConstraintSolverSetter>([](sofa::core::objectmodel::Base* object)
    {
        return py::cast(dynamic_cast<ConstraintSolverSetter*>(object));
    });
}

}
