cmake_minimum_required(VERSION 3.12)
project(CondensedFEMModel VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

sofa_find_package(SoftRobots REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)

# List all files
set(CondensedFEMModel_SRC_DIR src/${PROJECT_NAME})

set(HEADER_FILES
    ${CondensedFEMModel_SRC_DIR}/config.h.in
    ${CondensedFEMModel_SRC_DIR}/constraint/ConstraintPoint.h
    ${CondensedFEMModel_SRC_DIR}/engine/ConstraintSolverSetter.h
)
set(SOURCE_FILES
    ${CondensedFEMModel_SRC_DIR}/init.cpp
    ${CondensedFEMModel_SRC_DIR}/constraint/ConstraintPoint.cpp
    ${CondensedFEMModel_SRC_DIR}/constraint/ConstraintPoint.inl
    ${CondensedFEMModel_SRC_DIR}/engine/ConstraintSolverSetter.cpp
    ${CondensedFEMModel_SRC_DIR}/engine/ConstraintSolverSetter.inl
)
set(README_FILES
    README.md)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

target_link_libraries(${PROJECT_NAME} SoftRobots)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Constraint.Lagrangian.Solver)

sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python")
set(SP3_PYTHON_PACKAGES_DIRECTORY "python3/site-packages")
add_subdirectory(${CondensedFEMModel_SRC_DIR}/binding)


# Create package Config, Version & Target files.
# Deploy the headers, resources, scenes & examples.
# Set the plugin 'relocatable' if built within SOFA.
# --> see SofaMacros.cmake
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
)

sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "examples/python3/")
