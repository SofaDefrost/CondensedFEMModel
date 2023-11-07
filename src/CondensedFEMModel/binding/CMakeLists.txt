project(CondensedFEMModelBindings)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Binding_CondensedFEMModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Module_CondensedFEMModel.cpp
)

set(HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Binding_CondensedFEMModel.h
)

set(PYBIND11_PYTHON_VERSION 3.7)
find_package(Python ${PYBIND11_PYTHON_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 2.3 CONFIG QUIET REQUIRED)
find_package(SofaPython3 REQUIRED COMPONENTS Plugin Bindings.Sofa)

SP3_add_python_module(
    TARGET       ${PROJECT_NAME}
    PACKAGE      CondensedFEMModel
    MODULE       CondensedFEMModel
    DESTINATION  Sofa
    SOURCES      ${SOURCE_FILES}
    HEADERS      ${HEADER_FILES}
    DEPENDS      SofaPython3::Plugin SofaPython3::Bindings.Sofa CondensedFEMModel

)
message("-- SofaPython3 bindings for CondensedFEMModel will be created.")
