# -*- coding: utf-8 -*-
"""Base config class to save config data.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

import os
import gmsh
import tempfile
import hashlib
import shutil


class BaseConfig(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.scene_name = model_name
        self.generation_script_name = "Generation"
        self.n_robot = 1

        self.is_inverse = False
        self.init_model_parameters()
        self.scene_config = {"inverseMode": self.is_inverse,
                             "goalPos": None,
                             "is_force": False}

        # Specific attribute for managing evolving design data acquisition
        self.in_data_acquisition_loop = False
        self.is_direct_control_sampling = True
        self.useRigidGoal = False

        # Specific attribute for managing Multithreading feature for evolving designs
        self.base_meshes_path = os.path.dirname(os.path.abspath(__file__)) + '/../Models/' + self.model_name + '/Meshes/'
        self.meshes_path = os.path.dirname(os.path.abspath(__file__)) + '/../Models/' + self.model_name + '/Meshes/Cache/'


    def get_scene_name(self):
        """
        Return the name of the model SOFA simulation scene

        Outputs
        ----------
        name : str
            Name of the SOFA scene
        """
        return self.scene_name


    #########################################################
    ### Methods for managing traveling in actuation space ###
    #########################################################
    def get_actuators_variables(self):
        """
        Return a dictionnary {key = name_act, value = [init_value, min_value, max_value]} for each actuator variable.

        Outputs
        ----------
        name_act : str
            Name of the actuator. It must be the same in the associated SOFA scene.
        init_value : float
            Initial value for the actuator.
        min_value : float
            Minimum value for the actuator.
        max_value : float
            Maximum value for the actuator.
        """
        return {}

    def get_contacts_variables(self):
        """
        Return a dictionnary {key = name_cont, value = [init_value, min_value, max_value]} for each contact variable.
        There is 1 to list of values to provide depending on the dimension considered for the contact point (1D, 2D or 3D).

        Outputs
        ----------
        name_cont : str
            Name of the contact var. It must be the same in the associated SOFA scene.
            We use the PointEffector actuator component for representing contacts in the SOFA scene.
        init_value : float
            Initial values for the contact displacement.
        min_value : float
            Minimum values for the contact displacement.
        max_value : float
            Maximum values for the contact displacement.
        """
        return {}

    def get_inverse_variables(self):
        """
        Return a dictionnary {key = name_inv, value = [init_value, min_value, max_value]} for each inverse variable.
        The inverse variable is the variable defining the position / orientation of the effector in an inverse problem.

        Outputs
        ----------
        name_inv : str
            Name of the inverse var. We have to handle it in the controler of the SOFA scene.
        init_value : float
            Initial values for the variable (position, orientation).
        min_value : float
            Minimum values for the variable (position, orientation).
        max_value : float
            Maximum values for the variable (position, orientation).
        """
        return {}

    def interpolate_variables(self, normalized_values, var_type = "actuation"):
        """
        Compute bounds interpolation of [0,1] to [min_value, max_value] for a list of actuation values.
        This method should be reimplemented if using a different sampling strategy.
        ----------
        Parameters
        ----------
        normalized_values: list of float
            Normalized values for the actuator in [0,1].
        var_type: string in {"actuation", "contact"}
            Specify if the variable is an actuation or contact variable.
        ----------
        Outputs
        ----------
        values: list of float
            Value for the actuator in [min_value, max_value].
        """
        if var_type == "contact":
            variables = list(self.get_contacts_variables().values())
        elif var_type == "design": 
            variables = list(self.get_design_variables().values())
        elif var_type == "inverse":
            variables = list(self.get_inverse_variables().values())
        else: #Actuation by default
            variables = list(self.get_actuators_variables().values())
        values = [normalized_values[i] * (variables[i][2] - variables[i][1]) + variables[i][1] for i in range(len(normalized_values))]
        return values


    ######################################################
    ### Methods for managing traveling in design space ###
    ######################################################
    def init_model_parameters(self):
        """
        This function implement initialization of model parameters
        """
        pass

    def get_design_variables(self):
        """
        Return a dictionnary {key = name, value = [value, min_value, max_value]} for each design variable.
        ----------
        Outputs
        ----------
        name : str
            Name of the design variable.
        value : float
            Value for the design variable.
        min_value : float
            Minimum value for the design variable.
        max_value : float
            Maximum value for the design variable.
        """
        return {}


    def set_design_variables(self, new_values):
        """
        Set new values for design variables.
        ----------
        Inputs
        ----------
        new_values: list of float
            New value for each design variables. The values are in the same order as in get_design_variables().
        """
        for i, name in enumerate(self.get_design_variables().keys()):
            if new_values[i] >= self.get_design_variables()[name][1] and new_values[i] <= self.get_design_variables()[name][2]:
                setattr(self, name, new_values[i])
            else:
                print("Error: assigned new value for design variable are out of bounds.")


    #############################################
    ### Methods for managing data acquisition ###
    #############################################
    def get_n_sampling_variables(self):
        """
        Return the number of variables used for sampling.
        By default this number is the number of constraint.
        This method should be reimplemented if using a different sampling strategy.

        Outputs
        ----------
        n_sampling_vars : int
            Number of samplign variables.
        """
        return len(self.get_actuators_variables()) + len(self.get_contacts_variables()) + len(self.get_design_variables())


    #######################################
    ### Methods for managing simulation ###
    #######################################
    def get_scene_name(self):
        """
        Return the name of the model SOFA simulation scene

        Outputs
        ----------
        name : str
            Name of the SOFA scene
        """
        return self.scene_name

    def get_scene_config(self):
        """
        Return a configuration dictionnary for the scene.

        Outputs
        ----------
        scene_config : dic
            Configuration dictionnary of the scene (keys = name of the parameters, value = the
            value of the parameters).
        """
        return self.scene_config

    def set_scene_config(self, config):
        """
        Update the scene_config with the config.

        Parameters
        ----------
        config : dic
            Configuration dictionnary to include in scene_config.
        """
        self.scene_config.update(config)

    def set_is_inverse(self):
        """
        Set the scene in inverse mode.
        """
        self.is_inverse = True
        self.set_scene_config({"inverseMode": self.is_inverse})
        
    def set_action_type(self, is_force = False):
        """
        Set the value type for the actuators in the scene
        
        Parameters:
        -----------
            is_force: bool
                Wether actuators are controlled in force or displacement.
        """
        self.set_scene_config({"is_force": is_force})
    
    
    def set_goalPos(self, pos):
        """
        Set the goalPos of the scene.

        Parameters:
        -----------
            pos: list of float
                The new position of the goal.
        """
        self.set_scene_config({"goalPos": pos})

    @staticmethod
    def get_n_eq_dt(self):
        """
        Return the number of dt step to reach equilibrium.
        Is usefull for robots subjets to other than actuation/collision forces such as gravity.

        Outputs:
        -----------
            n_eq_dt: int
                The number of dt steps.
        """
        return None

    @staticmethod
    def get_n_dt(self):
        """
        Return the number of dt step for a simulation evaluation.

        Outputs:
        -----------
            n_dt: int
                The number of dt steps.
        """
        return None

    def get_post_sim_n_eq_dt(self):
        """
        Return the number of dt step to wait after simulation.
        This number is 0 by default.

        Outputs:
        -----------
            post_sim_n_eq_dt: int
                The number of dt steps to wait after simulation.
        """
        return 10

    @staticmethod
    def get_trajectory(self):
        """
        Return a trajectory to be performed by the robot.

        Outputs:
        -----------
            goals: list of list of arrays
                List of successive goals describing a trajectory of the robot.
        """
        return None
    

    #######################################################################
    ###### Functions for managing Design Optimization Multithreading ######
    #######################################################################
    # An adaptation of the xshape library from the SoftRobot.DesignOptimization toolbox: https://github.com/SofaDefrost/SoftRobots.DesignOptimization

    #@staticmethod
    def manage_temporary_directories(self):
        """
        Check if the cache directories need to be emptied.
        """
        # Create mesh directory if not already done
        if not os.path.exists(self.base_meshes_path):
            print("Creating the {0} directory".format(self.base_meshes_path))
            os.mkdir(self.base_meshes_path)            

        # Crate cache directory if not already done
        if not os.path.exists(self.base_meshes_path + "/Cache/"):
            print("Creating the {0} directory to cache mesh generation data".format(self.base_meshes_path + "/Cache/"))
            os.mkdir(self.base_meshes_path + "/Cache/")      

        # Check that the Cache directory is not too big
        size = 0
        file = 0
        for ele in os.scandir(self.base_meshes_path + "/Cache/"):
            size+=os.path.getsize(ele)
            file+=1
        size = size/(1024*1024)
        if size > 1000:
            print("Temporary directory is in: "+self.base_meshes_path + "/Cache/")
            print("                     file: "+str(file))
            print("                     size: "+str(int(size))+" Mb")
            print("The cache directory is too big...  please consider cleaning")

    #@staticmethod
    def get_unique_filename(self, generating_function):
        """
        Get the unique name of a geometry using hashmap.
        ----------
        Inputs
        ----------
        generating_function: func
            Link to the gmsh generating function.
        ----------
        Outputs
        ----------
        hashed_name: string
            An unique hashed name for the generated geometry.
        """
        temporary_file = tempfile.NamedTemporaryFile(suffix='.geo_unrolled')
        temporary_file.close()
        gmsh.write(temporary_file.name)
        result = hashlib.md5(open(temporary_file.name).read().encode())

        md5digest=result.hexdigest()

        return generating_function.__name__+ "_" + md5digest

    #@staticmethod
    def get_mesh_filename(self, mode, refine, generating_function, **kwargs):
        """
        Get the full hashed name of a mesh.
        ----------
        Inputs
        ----------
        mode: string in {Surface, Volume}
            Mesh file mode.
        refine: boolean
            Indicate if we shoudl refien the mesh.
        generating_function: func
            Link to the gmsh generating function.
        
        **kwargs: args
            Arguments for the mesh generation.
        ----------
        Outputs
        ----------
        full_filename: string
            The full path to the generated mesh.
        """
        self.manage_temporary_directories()
        gmsh.initialize()
        # Silence gmsh so by default nothing is printed
        gmsh.option.setNumber("General.Terminal", 0)
        id = generating_function(**kwargs)
        gmsh.model.occ.synchronize()
        filename = self.get_unique_filename(generating_function)
        if mode == "Step":
            full_filename = os.path.join(self.meshes_path, filename+".step") 
        elif mode == "Surface":
            full_filename = os.path.join(self.meshes_path, filename+"_surface.stl")   
        elif mode == "Volume":
            full_filename = os.path.join(self.meshes_path, filename+"_volume.vtk") 
        if not os.path.exists(full_filename):
            # When we are generating the mesh, it is better to know something is happening so let's reactive the printed messages
            gmsh.option.setNumber("General.Terminal", 1)
            if mode == "Surface":
                gmsh.model.mesh.generate(2)
            elif mode == "Volume":
                gmsh.model.mesh.generate(3)
            if refine:
                gmsh.model.mesh.refine()
            gmsh.write(full_filename)
        gmsh.finalize()
        return full_filename
    
    def save(self, source_filename, as_filename):
        """
        Save a gmsh geometry in a file with a known filename.
        ----------
        Inputs
        ----------
        source_filename: string
            Path to the source filename. 
            A good use is here to provide the result of get_mesh_filename as input.
        as_filename: string
            Path to the copied file name.
        """
        return shutil.copy(source_filename, as_filename)
    
    def show(self, generating_function, **kwargs):
        """
        Show a generated gmsh geoemtry. 
        Usefull for debuguing.
        ----------
        Inputs
        ----------
        generating_function: func
            Link to the gmsh generating function.
        as_filename: string
            Path to the copied file name.
        """
        self.manage_temporary_directories()
        gmsh.initialize()
        # Silence gmsh so by default nothing is printed
        gmsh.option.setNumber("General.Terminal", 0)
        id = generating_function(**kwargs)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()
