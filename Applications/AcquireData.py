# -*- coding: utf-8 -*-
"""Main file to launch script (learning, data acquisition, applications).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 05 2022"

# System libs
import sys
import pathlib
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


# Local libs
import time
import numpy as np
import pickle
import copy
import os
from Libraries.database import *

# SOFA libs
import Sofa
import SofaRuntime
SofaRuntime.importPlugin("SofaPython3")
from Sofa import SofaConstraintSolver


#############################################################
################## Data Acquisition Scheme ##################
#############################################################
def acquire_data(config, n_train_samples=100000, ratio_test_train=0.25, n_process=1, evolve_design=False, use_GUI=False, train_sampling_strategy="Grid"):
    """ Function to manage data acquisition for learning

    Parameters
    ----------
    config : Config
        Config describing the model
    n_train_samples: int
        Maximum number of train samples
    ratio_test_train: float
        The number of test samples is given by ratio_test_train * n_train_sample
    n_process: int
        Number of process used for data acquisition.
    evolve_design: Boolean
        To evolve a parameterized design during data generation or not
    use_GUI: boolean
        To visualize SOFA scene in a GUI
    """
    
    # Do not display simulation if too much process
    if n_process > 1:
        import multiprocessing
    if n_process > 1:
        use_GUI = False

    # Create database if not already done
    database_path = pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + "/../Results/data.db")
    if not pathlib.Path.exists(database_path):
        create_tables()
    connect_db()

    # Add the model in the database
    add_model(config.model_name)

    # Manage data acquisition
    sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
    scene_lib = importlib.import_module("Models." + config.scene_name + "." + config.model_name)    


    def complete_database(sampled_vars, id_sampling_stats, n_samples, sampling_strategy, is_train):
        """ Function to manage simulation and database filling.

            Parameters
            ----------
            sampled_vars : list of list of floats
                Values for actuation and contacts.
            id_sampling_stats: int
                Reference towards the corresponding sampling stats                    
            sampling_strategy: str in {Random, Grid}
                Method for data sampling.
            n_samples: int
                Number of data samples to compute.
            is_train: boolean
                Check if data is for train or test set.
            """


        def add_simulation_line_direct(sample_vars):
            """ Function to add a new line in database, with a direct sampling of the actuators / contacts

            Parameters
            ----------
            sample_vars: list of floats in range [0,1]
                Sampled values for all samplign variables in the order actuation/contact/design
            """
            # Separate constraint values from design variables
            n_act = len(config.get_actuators_variables())
            n_sample_constraint_size = len(sample_vars) - len(config.get_design_variables())
            sample_actuation = sample_vars[:n_act]
            sample_contact = sample_vars[n_act:n_sample_constraint_size]
            sample_design = sample_vars[n_sample_constraint_size:]

            # Compute real values for constraints
            interpolated_actuation = config.interpolate_variables(sample_actuation, var_type = "actuation")
            interpolated_contact = config.interpolate_variables(sample_contact, var_type = "contact")
            interpolated_constraints = interpolated_actuation + interpolated_contact

            # Update config with real values for contacts
            interpolated_design = []
            if len(sample_design) != 0:
                interpolated_design = config.interpolate_variables(sample_design, var_type = "design")

            # Direct simulation of sampled variables
            W_0, dfree_0, actuation_state, effectors_state, W, dfree = direct_simulation(config, scene_lib, interpolated_actuation, interpolated_contact, interpolated_design)

            # Add simulated results to database
            update_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = n_samples)
            add_simulation_results(config.model_name, id_sampling_stats, pickle.dumps(interpolated_design), 
                                pickle.dumps(W_0), pickle.dumps(dfree_0), pickle.dumps(interpolated_constraints), 
                                pickle.dumps(actuation_state), pickle.dumps(effectors_state), pickle.dumps(W), 
                                pickle.dumps(dfree), is_train = is_train)
            

        def add_simulation_line_inverse(sample_vars):
            """ Function to add a new line in database, with inverse sampling of the actuators / contacts 

            Parameters
            ----------
            sample_vars: list of floats in range [0,1]
                Sampled values for all samplign variables used in the inverse sampling model.
            """
            # Separate constraint values from design variables
            n_inverse_variables = len(config.get_inverse_variables())
            sample_inverse = sample_vars[:n_inverse_variables]
            sample_design = sample_vars[n_inverse_variables:]
            interpolated_inverse = config.interpolate_variables(sample_inverse, var_type="inverse")

            interpolated_design = []
            if len(sample_design) != 0:
                interpolated_design = config.interpolate_variables(sample_design, var_type="design")

            # Inverse simulation of sampled variables
            W_0, dfree_0, actuation_state, effectors_state, W, dfree = inverse_simulation(config, scene_lib, interpolated_inverse, interpolated_design)

            # Add simulated results to database
            for id_robot in range(config.nb_robot):
                update_sampling_stats(config.model_name, sampling_strategy=sampling_strategy, n_samples=n_samples)
                add_simulation_results(config.model_name, id_sampling_stats, pickle.dumps(interpolated_design),
                                       pickle.dumps(W_0[0]), pickle.dumps(dfree_0[0]),
                                       pickle.dumps(interpolated_inverse), pickle.dumps(actuation_state[id_robot]),
                                       pickle.dumps(effectors_state[id_robot]), pickle.dumps(W[id_robot]), pickle.dumps(dfree[id_robot]),
                                   is_train=is_train)

                # add_simulation_results(config.model_name, id_sampling_stats, pickle.dumps(interpolated_design),
                #                        pickle.dumps(W_0[id_robot]), pickle.dumps(dfree_0[id_robot]),
                #                        pickle.dumps(interpolated_inverse), pickle.dumps(actuation_state[id_robot]),
                #                        pickle.dumps(effectors_state[id_robot]), pickle.dumps(W[id_robot]), pickle.dumps(dfree[id_robot]),
                #                    is_train=is_train)



        if len(sampled_vars) != 0:
            null_action = (len(config.get_actuators_variables()) + len(config.get_contacts_variables())) * [0]

            try:
                if config.is_direct_control_sampling:
                    add_simulation_line = add_simulation_line_direct
                else:
                    add_simulation_line = add_simulation_line_inverse

                if n_process == 1:
                    for i, sample_vars in enumerate(sampled_vars):
                        print("[INFO] >> Iteration:", i, "/", len(sampled_vars))
                        add_simulation_line(sample_vars)

                elif n_process > 1:
                    current_id = 0
                    while current_id < len(sampled_vars):
                        jobs = []
                        for i in range(0, n_process):
                            if i+current_id < len(sampled_vars):
                                print("[INFO] >> Iteration:", i+current_id, "/", len(sampled_vars))
                                process = multiprocessing.Process(target=add_simulation_line, args=([sampled_vars[i+current_id]]))
                                jobs.append(process)
                        current_id+= n_process

                        if jobs != []:
                            for j in jobs:
                                j.start()

                            # Ensure all of the processes have finished
                            for j in jobs:
                                j.join()


            except:
                print("[ERROR] >> The scene did crash")

                # Update sample actuation
                registered_sampling_state = query_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = n_samples)
                sampling_stats_id = registered_sampling_state[0]["id"]
                n_next_sample = registered_sampling_state[0]["n_curr_sample"] + 1
                sampled_vars = sampled_vars[n_next_sample::]


    path = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute())))

    ##### Train Set #####
    # Sample actuation space
    sampling_strategy = train_sampling_strategy

    file_name = pathlib.Path(path + "/temporary_files/" + config.model_name + "_" + \
        sampling_strategy + "_" + str(n_train_samples) + ".pickle")
    if pathlib.Path.exists(file_name):
        with open(file_name, 'rb') as f:
            sampled_vars = pickle.load(f)
    else:
        sampled_vars = sample_vars(config, method=sampling_strategy, n_samples = n_train_samples)

    train_n_samples = len(sampled_vars)
    registered_sampling_state = query_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = train_n_samples)
    if len(registered_sampling_state) == 0: # Create a new data acquisition
        train_sampling_stats_id = add_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = train_n_samples)
    else: # Restart from a previously started data acquisition
        train_sampling_stats_id = registered_sampling_state[0]["id"]
        n_next_sample = registered_sampling_state[0]["n_curr_sample"] + 1
        sampled_vars = sampled_vars[n_next_sample::]

    # Simulate and complete database
    complete_database(sampled_vars, id_sampling_stats = train_sampling_stats_id, n_samples = train_n_samples, sampling_strategy = sampling_strategy, is_train = True)
    os.remove(path + "/temporary_files/" + config.model_name + "_" + \
        sampling_strategy + "_" + str(n_train_samples) + ".pickle")

    ##### Test Set #####
    # Sample actuation space
    sampling_strategy = "Random"
    test_n_samples = int(ratio_test_train * train_n_samples)

    file_name = pathlib.Path(path + "/temporary_files/" + config.model_name + "_" + \
        sampling_strategy + "_" + str(test_n_samples) + ".pickle")
    if pathlib.Path.exists(file_name):
        with open(file_name, 'rb') as f:
            sampled_vars = pickle.load(f)
    else:
        sampled_vars = sample_vars(config, method=sampling_strategy, n_samples = test_n_samples)
    registered_sampling_state = query_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = test_n_samples)
    if len(registered_sampling_state) == 0: # Create a new data acquisition
        test_sampling_stats_id = add_sampling_stats(config.model_name, sampling_strategy = sampling_strategy, n_samples = test_n_samples)
    else: # Restart from a previously started data acquisition
        test_sampling_stats_id  = registered_sampling_state[0]["id"]
        n_next_sample = registered_sampling_state[0]["n_curr_sample"] + 1
        sampled_vars = sampled_vars[n_next_sample::]

    # Simulate and complete database
    complete_database(sampled_vars, id_sampling_stats = test_sampling_stats_id , n_samples = test_n_samples, sampling_strategy = sampling_strategy, is_train = False)
    os.remove(path + "/temporary_files/" + config.model_name + "_" + \
        sampling_strategy + "_" + str(test_n_samples) + ".pickle")
    disconnect_db()


###########################################################
################## Simulation Assessment ##################
###########################################################
def direct_simulation(config, scene_lib, actuation_vars, contact_vars, design_vars):
    """ Direct simulation from variables describing the actuation/contact/design space.
    -------
    Inputs
    -------
    config : Config
        Config describing the model
    scene_lib: python library
        Link to the SOFA scene library for the considered config.
    actuation_vars: list of floats
        Actuator displacement values in their usual range.
    contact_vars: list of floats    
        Contact displacement values in their usual range.
    design_vars: list of floats
        Design variables values in their usual range.
    -------
    Outputs
    -------
    W_0, dfree_0, actuation_state, effectors_state, W, dfree: numpy arrays
        The mechanical matrices sampled from the scene.
    """   

    # Update config with real values for design variables
    copy_config = config
    if len(design_vars) != 0:
        copy_config.set_design_variables(design_vars)

    # Get config time settings
    n_dt = config.get_n_dt()
    n_eq_dt = config.get_n_eq_dt()
    post_sim_n_eq_dt = config.get_post_sim_n_eq_dt() 

    # Init SOFA scene
    root = Sofa.Core.Node("root")
    scene_lib.createScene(root, copy_config)
    Sofa.Simulation.init(root)

    # Animate without actuation to reach equilibrium
    null_action = (len(config.get_actuators_variables()) + len(config.get_contacts_variables())) * [0]
    root.Controller.apply_actions(null_action)
    for step in range(n_eq_dt):
        Sofa.Simulation.animate(root, root.dt.value)
        time.sleep(root.dt.value)
    W_0 = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
    dfree_0 = copy.deepcopy(root.Controller.get_dfree())

    # Apply actuation / contact displacement
    constraints_vars = actuation_vars + contact_vars
    for step in range(n_dt + post_sim_n_eq_dt):
        constraints_vars_step = [min((step + 1) * v/(n_dt), v) for v in constraints_vars] # Apply gradually action
        root.Controller.apply_actions(constraints_vars_step)
        Sofa.Simulation.animate(root, root.dt.value)
        time.sleep(root.dt.value)

    actuation_state = copy.deepcopy(root.Controller.get_actuators_state()) # Contact points are handled as actuators in the SOFA scene
    effectors_state = copy.deepcopy(root.Controller.get_effectors_state())
    W = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
    dfree = copy.deepcopy(root.Controller.get_dfree())
    print(">> Simulation done")

    # Reset simulation
    Sofa.Simulation.reset(root)

    return W_0, dfree_0, actuation_state, effectors_state, W, dfree


def inverse_simulation(config, scene_lib, inverse_vars, design_vars):
    """ Direct simulation from variables describing both the inverse sampling and design space.
    -------
    Inputs
    -------
    config : Config
        Config describing the model
    scene_lib: python library
        Link to the SOFA scene library for the considered config.
    inverse_vars: list of floats
        Displacement values for variables of the inverse sampling scheme.
    design_vars: list of floats
        Design variables values in their usual range.
    -------
    Outputs
    -------
    W_0, dfree_0, actuation_state, effectors_state, W, dfree: numpy arrays
        The mechanical matrices sampled from the scene.
    """ 

    # Update config with real values for design variables
    if len(design_vars) != 0:
        copy_config.set_design_variables(design_vars)

    # Get config time settings
    n_dt = config.get_n_dt()
    n_eq_dt = config.get_n_eq_dt()
    post_sim_n_eq_dt = config.get_post_sim_n_eq_dt()

    # Update config with inverse sampling goal
    copy_config = config
    copy_config.set_goalPos(inverse_vars)

    # Init SOFA scene
    root = Sofa.Core.Node("root")
    scene_lib.createScene(root, copy_config)
    Sofa.Simulation.init(root)

    # Animate without actuation to reach equilibrium
    for step in range(n_eq_dt):
        Sofa.Simulation.animate(root, root.dt.value)
        time.sleep(root.dt.value)
    W_0 = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
    dfree_0 = copy.deepcopy(root.Controller.get_dfree())

    # Apply actuation / contact displacement
    for step in range(n_dt + post_sim_n_eq_dt):
        Sofa.Simulation.animate(root, root.dt.value)
        time.sleep(root.dt.value)

    actuation_state = copy.deepcopy(root.Controller.get_actuators_state())  # Contact points are handled as actuators in the SOFA scene
    effectors_state = copy.deepcopy(root.Controller.get_effectors_state())
    W = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
    dfree = copy.deepcopy(root.Controller.get_dfree())
    print(">> Simulation done")

    # Reset simulation
    Sofa.Simulation.reset(root)

    return W_0, dfree_0, actuation_state, effectors_state, W, dfree


##############################################################
################## Offline Sampling Schemes ##################
##############################################################
def sample_vars(config, method="Grid", n_samples = 10000):
    """ Function to manage data acquisition for learning
    ----------
    Parameters
    ----------
    config : Config
        Config describing the model.
    method: str in {Random, Grid}
        Method for data sampling.
        Random: sample points at random.
        Grid: exhaustively sample actuation space.
        SH: Scrambled Halton sampling i.e. low discrepancy sequence with scrambling to ameliorate Halton sequence by limitating severe striping artifacts
        LHS: Latin Hypercube Sampling generates a set of samples such that each sample point is formed by randomly selecting one coordinate value from a grid from each dimension
        LCVT: Latinized Centroidal Voronoi Tessellation as described in "Smart sampling and incremental function learning for very large high dimensional data"
    n_samples: int
        Number of samples. This input number is updated to obtain an homogeneous Grid strategy.
    ----------
    Output
    ----------
    sampled_vars: list of lists of float
        Samples for actuation/contact variables
    """

    n_vars = config.get_n_sampling_variables()

    if method == "Random":
        print(">>> Start Random sampling ...")
        sampled_vars = [0] * n_samples
        for i in range(n_samples):
            sampled_vars[i] = [np.random.uniform(0,1) for j in range(n_vars)]
        print("... End <<<")

    elif method == "Grid":
        print(">>> Start Grid sampling ...")
        n_sample_interval = max(2, int(np.exp(np.log(n_samples)/n_vars)))
        sampling_01 = list(np.linspace(0, 1, n_sample_interval))
        all_sampling01 = [sampling_01 for i in range(n_vars)]
        sampled_vars = [list(x) for x in np.array(np.meshgrid(*all_sampling01)).T.reshape(-1,len(all_sampling01))]
        print("... End <<<")

    elif method == "LHS":
        print(">>> Start LHS sampling ...")
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_vars)
        sampled_vars = sampler.random(n=n_samples)
        print("... End <<<")

    elif method == "SH":
        from scipy.stats import qmc
        print(">>> Start ScrambledHalton sampling ...")
        sampler = qmc.Halton(d=n_vars, scramble=True)
        sampled_vars = sampler.random(n=n_samples)
        print("... End <<<")

    elif method == "LCVT":
        print(">>> Start LCVT sampling ...")
        from scipy.stats import qmc
        from sklearn.cluster import KMeans

        # Create Halton sequence as initial populationid_sampling
        # Halton sequence are low discrepancy i.e. the proportion of points in the sequence falling into an arbitrary set B is close to proportional to the measure of B
        init_n_samples = 10 * n_samples
        sampler = qmc.Halton(d=n_vars, scramble=True)
        init_samples = sampler.random(n=init_n_samples)

        # Identify the n_samples generators of the initial distribution using Kmeans for computing centroids
        kmeans = KMeans(
            init="k-means++",
            algorithm="full", #lloyd
            n_clusters=n_samples,
            n_init=1,
            max_iter=100,
            tol=0.0001,
            )
        kmeans.fit(init_samples)
        centroids = kmeans.cluster_centers_
        samples = centroids.tolist()

        # Latinize the distribution
        for j in range(n_vars):
            # Reorder samples in increasing order regarding dimension j
            samples.sort(key=lambda x:int(x[j]))
            # Divide the range of values in the j-th dimension into n_samples equispaced bins
            bin_length = 1 / n_samples
            assiocated_bins_j = [samples[k][j] // bin_length for k in range(n_samples)]
            # Shift non self-contained data in their bin to a random position position in the correct bin
            for i in range(n_samples):
                if assiocated_bins_j[i] != i:
                    samples[i][j] = np.random.uniform(i*bin_length, (i+1)*bin_length)
        print("... End <<<") 

    else:
        print("[ERROR] >> The sampling strategie is not implemented yet.")
        exit(1)
    
    current_path = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute())))
    os.makedirs(current_path + "/temporary_files", exist_ok=True)
    with open(current_path + "/temporary_files/" + config.model_name + "_" + method + "_" + str(n_samples) + ".pickle", 'wb') as f:
        pickle.dump(sampled_vars, f, protocol=pickle.HIGHEST_PROTOCOL)

    return sampled_vars
