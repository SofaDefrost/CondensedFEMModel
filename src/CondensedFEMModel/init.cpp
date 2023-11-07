#include <CondensedFEMModel/config.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

extern "C" {
    CondensedFEMModel_API void initExternalModule();
    CondensedFEMModel_API const char* getModuleName();
    CondensedFEMModel_API const char* getModuleVersion();
    CondensedFEMModel_API const char* getModuleLicense();
    CondensedFEMModel_API const char* getModuleDescription();
    CondensedFEMModel_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(CondensedFEMModel_VERSION);
}

const char* getModuleLicense()
{
    return "None";
}

const char* getModuleDescription()
{
    return "Predicting Compliance plugin";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}
