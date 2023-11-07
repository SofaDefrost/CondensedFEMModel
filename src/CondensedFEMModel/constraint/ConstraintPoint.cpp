#define SOFTROBOTS_CONSTRAINTPOINT_CPP

#include "ConstraintPoint.inl"
#include <sofa/core/ObjectFactory.h>

namespace CondensedFEMModel{

//----------- Position Force Constraint  Resolution --------------
PositionForceConstraintResolution::PositionForceConstraintResolution(double fx, double fy, double fz, unsigned int nbLines)
    : sofa::core::behavior::ConstraintResolution(nbLines)
    , nbLines(nbLines), mfx(fx), mfy(fy), mfz(fz)
{ }


void PositionForceConstraintResolution::resolution(int line, double** w, double* d, double* lambda, double* dfree)
{
    SOFA_UNUSED(dfree);
    SOFA_UNUSED(w);
    SOFA_UNUSED(d);
    
    lambda[line] = mfx;
    lambda[line+1] = mfy;
    lambda[line+2] = mfz;
}


//---------- Position Displacement Constraint Resolution ------------
PositionDisplacementConstraintResolution::PositionDisplacementConstraintResolution(double x, double y, double z, unsigned int nbLines)
    : sofa::core::behavior::ConstraintResolution(nbLines)
            , nbLines(nbLines), mx(x), my(y), mz(z)
    { }

void PositionDisplacementConstraintResolution::init(int line, double** w, double * lambda)
{
    SOFA_UNUSED(lambda);
    m_wActuatorActuator_x = w[line][line];
    m_wActuatorActuator_y = w[line+1][line+1];
    m_wActuatorActuator_z = w[line+2][line+2];
}


void PositionDisplacementConstraintResolution::resolution(int line, double** w, double* d, double* lambda, double* dfree)
{
    SOFA_UNUSED(dfree);
    SOFA_UNUSED(w);

    // da=Waa*(lambda_a) + Sum Wai * lambda_i  = m_imposedDisplacement
    // d is the cumulative sum
    lambda[line] -= (d[line]-mx) / m_wActuatorActuator_x;
    lambda[line+1] -= (d[line+1]-my) / m_wActuatorActuator_y;
    lambda[line+2] -= (d[line+2]-mz) / m_wActuatorActuator_z;

}




template<>
void ConstraintPoint<sofa::defaulttype::Rigid3Types>::normalizeDirections()
{
    VecDeriv directions;
    directions.resize(6);
    for(unsigned int i=0; i<6; i++)
    {
        directions[i] = d_directions.getValue()[i];
        Vec<3, Real> vector1 = Vec<3, Real>(directions[i][0],directions[i][1],directions[i][2]);
        Vec<3, Real> vector2 = Vec<3, Real>(directions[i][3],directions[i][4],directions[i][5]);
        vector1.normalize();
        vector2.normalize();
        directions[i] = Deriv(vector1,vector2);
    }
    d_directions.setValue(directions);
}


template<>
void ConstraintPoint<sofa::defaulttype::Rigid3Types>::draw(const VisualParams* vparams)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return;

    if (!vparams->displayFlags().getShowInteractionForceFields())
        return;

    vector<sofa::type::Vec3> points;
    VecCoord positions = m_state->read(sofa::core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<d_indices.getValue().size(); i++)
    {
        points.push_back(positions[d_indices.getValue()[i]].getCenter());
        points.push_back(d_effectorGoalPositions.getValue()[i].getCenter());
    }
    vparams->drawTool()->drawPoints(points,10.0f,RGBAColor(0.,1.,0.,1.));
}



////////////////////////////////////////////    FACTORY    //////////////////////////////////////////////
using namespace sofa::helper;

// Registering the component
// see: http://wiki.sofa-framework.org/wiki/ObjectFactory
// 1-RegisterObject("description") + .add<> : Register the component
// 2-.add<>(true) : Set default template

volatile int ConstraintPointClass = sofa::core::RegisterObject("This component is used to describe one or several desired positions "
                                                 "of points of a model, that will be reached by acting on chosen actuator(s).")
                .add< ConstraintPoint<sofa::defaulttype::Vec3Types> >(true)
                .add< ConstraintPoint<sofa::defaulttype::Rigid3Types> >()

        ;

////////////////////////////////////////////////////////////////////////////////////////////////////////

// Force template specialization for the most common sofa floating point related type.
// This goes with the extern template declaration in the .h. Declaring extern template
// avoid the code generation of the template for each compilation unit.
// see: http://www.stroustrup.com/C++11FAQ.html#extern-templates
template class SOFA_SOFTROBOTS_API ConstraintPoint<sofa::defaulttype::Vec3Types>;
template class SOFA_SOFTROBOTS_API ConstraintPoint<sofa::defaulttype::Rigid3Types>;


} // namespace CondensedFEMModel