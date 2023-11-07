""" This script implements methods for ploting results obtained from surrogate-assisted optimization of design parameters"""
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Apr 12 2023"


# Local libs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': '25',
         'axes.titlesize':'25',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
import importlib
from Applications.DesignOptimization import init_loss, load_networks, init_problem, prepare_data, compute_loss


def main():
    ### Scenario choice
    SCENARIO = "FINGER_DESIGN_MULTI_OBJS_Pareto" # "FINGER_DESIGN_MULTI_OBJS_GRID" #"FINGER_DESIGN_ANGLE_LANDSCAPE" # "FINGER_MULTI_OBJS_LANDSCAPE" # "FINGER_DESIGN_CONTACT_LANDSCAPE" # "FINGER_DESIGN_CONTACT" #"FINGER_DESIGN_ANGLE_LANDSCAPE" # "FINGER_DESIGN_ANGLE" # "FINGER_CALIB_LANDSCAPE" 
    
    ### Problem init
    if SCENARIO == "FINGER_CALIB_SINGLE_INIT":
        problem_name = "CalibrationMechanicalParams"
        n_step = 620
        learning_rate = 1e-2
        init_params = [[0.47, 5000]]
    
    elif SCENARIO == "FINGER_CALIB_MULTIPLE_INIT":
        problem_name = "CalibrationMechanicalParams"
        n_step = 620
        learning_rate = 1e-2
        init_params = [[0.47, 5000], [0.49, 7000], [0.45, 2000]]
        
    elif SCENARIO == "FINGER_CALIB_LANDSCAPE":
        problem_name = "CalibrationMechanicalParams"
        name_config = "FingerElasticityParams"
        baseline = None
        baseline_loss = None
               
    elif SCENARIO == "FINGER_LENGTH_ANGLE":
        problem_name = "DexterityObjective"
        n_step = 500
        learning_rate = 1e-2
        init_params = [[37.5]]
        
    elif SCENARIO == "FINGER_DESIGN_ANGLE":
        problem_name = "DexterityObjective"
        n_step = 4000
        learning_rate = 1e-1
        init_params = [[40, 21, 6.0]] 
        
    elif SCENARIO == "FINGER_DESIGN_ANGLE_LANDSCAPE":
        problem_name = "DexterityObjective"
        name_config = "FingerDesign"
        
        baseline = [38.0, 20.53, 6.11]
        baseline_loss = 0.9871
        
    elif SCENARIO == "FINGER_DESIGN_ANGLE_GRID":
        problem_name = "DexterityObjective"
        name_config = "FingerDesign"
        
        
    elif SCENARIO == "FINGER_DESIGN_CONTACT":
        problem_name = "ContactForceObjective"
        n_step = 1000
        learning_rate = 1e-1
        init_params = [[40, 21, 6.0]]
        
    elif SCENARIO == "FINGER_DESIGN_CONTACT_LANDSCAPE":
        problem_name = "ContactForceObjective"
        name_config = "FingerDesign"
        baseline = [41.69230769, 21.1049411, 5.92307692] # [41.15151968, 21.1049411 ,  5.6184284 ]
        baseline_loss = 0.68956622
    
    elif SCENARIO == "FINGER_DESIGN_CONTACT_GRID":
        problem_name = "ContactForceObjective"
        name_config = "FingerDesign"
        
    elif SCENARIO == "FINGER_MULTI_OBJS_LANDSCAPE":
        problem_name = "BendingAngleAndContactForceObjective"
        name_config = "FingerDesign"
        baseline = [41.21214606, 21.18435532,  5.47039068]
        baseline_loss = 2.2116
    
    elif SCENARIO == "FINGER_DESIGN_MULTI_OBJS_GRID":
        problem_name = "BendingAngleAndContactForceObjective"
        name_config = "FingerDesign"
        
    elif SCENARIO == "FINGER_DESIGN_MULTI_OBJS_Pareto":
        problem_name = "BendingAngleAndContactForceObjective"
        name_config = "FingerDesign"
        
        
    single_plot_objs = ["FINGER_CALIB_SINGLE_INIT", 
                        "FINGER_LENGTH_ANGLE",
                        "FINGER_DESIGN_ANGLE",
                        "FINGER_DESIGN_CONTACT"        
    ]
    
    fitness_landscape_plot_objs = ["FINGER_CALIB_LANDSCAPE",
                                   "FINGER_DESIGN_ANGLE_LANDSCAPE",
                                   "FINGER_DESIGN_CONTACT_LANDSCAPE",
                                   "FINGER_MULTI_OBJS_LANDSCAPE"
    ]
    
    best_init_grid_objs = [
        "FINGER_DESIGN_ANGLE_GRID",
        "FINGER_DESIGN_CONTACT_GRID",
        "FINGER_DESIGN_MULTI_OBJS_GRID"
    ]
    
    pareto_objs = [
        "FINGER_DESIGN_MULTI_OBJS_Pareto"
    ]
    
    
    ## Load optimization histories
    if SCENARIO in single_plot_objs or SCENARIO == "FINGER_CALIB_MULTIPLE_INIT":
        list_names_design_params = []
        list_loss_per_iter = []
        list_params_per_iter = []
        for j in range(len(init_params)):
            init_params_string = ""
            for i in range(len(init_params[j])):
                init_params_string += "_" + str(init_params[j][i]) + "p" + str(i)
            name_DO_results = problem_name + "_" + str(n_step) + "it_" + str(learning_rate) + "lr" + init_params_string    
            list_names_design_params.append(np.load("OptimizationHistories/" + name_DO_results + "_param_names.npy").tolist())
            list_loss_per_iter.append(np.load("OptimizationHistories/" + name_DO_results + "_loss_hist.npy").tolist())
            list_params_per_iter.append(np.load("OptimizationHistories/" + name_DO_results + "_params_hist.npy").tolist())
        
        
    ### Generate plots
    if SCENARIO in single_plot_objs:
        plot_obj_history(list_loss_per_iter[0])
        plot_params_history(list_names_design_params[0], list_params_per_iter[0])
    elif SCENARIO == "FINGER_CALIB_MULTIPLE_INIT":
        plot_multiple_obj_history(list_names_design_params[0], init_params, list_loss_per_iter)
        for i in range(len(list_names_design_params)):
            plot_params_history(list_names_design_params[i], list_params_per_iter[i])     
    elif SCENARIO in fitness_landscape_plot_objs:
        plot_objective_landscape(problem_name, name_config, n_samples = 200, baseline = baseline, baseline_loss = baseline_loss) 
    elif SCENARIO in best_init_grid_objs:
        grid_search_best(problem_name, name_config, n_samples = 5000)       
    elif SCENARIO in pareto_objs:
        grid_pareto(problem_name, name_config, n_samples = 300)
        


def plot_obj_history(loss_per_iter):
    """Plot objective history

    Args:
        loss_per_iter (list of float): Values for the fitness function accross optimization iterations.
    """
    x = list(range(len(loss_per_iter)))
    y = loss_per_iter
    plt.plot(x, y)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Function Values")
    plt.show()
    
def plot_multiple_obj_history(names_design_params, init_params, list_loss_per_iter, with_legend = True):
    """Plot objective history

    Args:
        names_design_params (list of strings): Names of each design variable.
        init_params (list of list of floats): Initial parameters values for each experiment.
        list_loss_per_iter (list of lists of float): Values for the fitness function accross optimization iterations for several experiments.
        with_legend (bool): Specify if a legend is used or not.
    """
    x = list(range(len(list_loss_per_iter[0])))
    for i in range(len(list_loss_per_iter)):
        y = list_loss_per_iter[i]
        y_label = ""
        for j in range(len(names_design_params)):
            if j != 0:
                y_label += ", "
            y_label += names_design_params[j] + " = " + str(init_params[i][j])
        plt.plot(x, y, '--', label = y_label)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Function Values")
    
    if with_legend:
        plt.legend()
    plt.show()
    
    
def plot_params_history(names_design_params, params_per_iter, with_legend = False):
    """Plot several parameters history on a single figure. 
    This function can be used with up to 3 parameters

    Args:
        names_design_params (list of strings): Names for each design variables.
        params_per_iter (list of lists of floats): List of list of values for each parameter for each optimization iteration.
        with_legend (bool): Specify if a legend is used or not.
    """
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt
    # List of available colors for plotting
    available_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # Initialize plot and set axes
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    if len(names_design_params) > 3 :
        print("Too much design parameters for design parameters convergence history. Exiting code...")
        exit()
    host.set_xlabel("Iterations")
    host.set_ylabel(names_design_params[0],  color = available_colors[0])
    if len(names_design_params) >= 2 :
        par1 = host.twinx()
        par1.set_ylabel(names_design_params[1], color = available_colors[1])
        par1.axis["right"].toggle(all=True)
    if len(names_design_params) == 3:
        par2 = host.twinx()
        offset = 60
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                                offset=(offset, 0))
        par2.axis["right"].toggle(all=True)    
        par2.set_ylabel(names_design_params[2], color = available_colors[2])
    # Plot values
    x = list(range(len(params_per_iter)))
    p1, = host.plot(x, [params_per_iter[i][0] for i in range(len(params_per_iter))], 
                    label=names_design_params[0],
                    color = available_colors[0])
    if len(names_design_params) >= 2 :
        p2, = par1.plot(x, [params_per_iter[i][1] for i in range(len(params_per_iter))], 
                        label=names_design_params[1],
                        color = available_colors[1])
    if len(names_design_params) == 3:    
        p3, = par2.plot(x, [params_per_iter[i][2] for i in range(len(params_per_iter))], 
                        label=names_design_params[2],
                        color = available_colors[2])
    if with_legend:
        host.legend()
    plt.draw()
    plt.show()
       
def plot_objective_landscape(problem_name, name_config, n_samples, baseline, baseline_loss):
    import itertools
    
    # Retrieve config
    config_lib = importlib.import_module("Models." + name_config + ".Config")
    config = config_lib.Config()
    name_design_vars = list(config.get_design_variables().keys())
        
    # Load the MLP and scalers
    pred_model, optimizer, data_scaling_params, data_scaling_MM = load_networks(config)
    loss_function = init_loss(problem_name) # Select corresponding loss
    
    # Init problem parameters
    if problem_name == "CalibrationMechanicalParams":
        list_s_a = [[0], [5], [10]]
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        simulation_values = init_problem(config, problem_name, list_s_a)
        values = [[act_disp for sublist in list_s_a for act_disp in sublist], simulation_values]
    elif problem_name == "DexterityObjective" or problem_name == "ContactForceObjective" or problem_name == "BendingAngleAndContactForceObjective":
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_s_a = [[10]] # Cable displacement used for evaluating the dexterity of the finger
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        values = [[act_disp for sublist in list_s_a for act_disp in sublist]]
        
    # Generate all pairs of design variables
    idx_design_vars = list(range(len(name_design_vars)))
    pairs = list(itertools.combinations(idx_design_vars, 2))
        
    # Generate samples in the design data range
    n_vars = 2
    n_sample_interval = max(2, int(np.exp(np.log(n_samples)/n_vars)))
    sampling_01 = list(np.linspace(0, 1, n_sample_interval))
    all_sampling01 = [sampling_01 for i in range(n_vars)]
    sampled_vars = [list(x) for x in np.array(np.meshgrid(*all_sampling01)).T.reshape(-1,len(all_sampling01))]
                            
    # Compute objective for each combo of two dimensions
    all_design_variables = []
    all_losses_per_sample = []
    for pair in pairs:
        design_variables = []
        loss_per_sample = []
        for sample_design_0_1 in sampled_vars:
            # Interpolate and retrieve real sampled design
            sorted_sample_design_0_1 = [0.5 for i in range(len(name_design_vars))]
            sorted_sample_design_0_1[pair[0]] = sample_design_0_1[0]
            sorted_sample_design_0_1[pair[1]] = sample_design_0_1[1]
            sample_design = config.interpolate_variables(sorted_sample_design_0_1, var_type = "design")
            
            # Replace non altered values with baseline values
            print("sample_design:", sample_design)
            print("baseline:", baseline)
            if baseline != None:
                for i in range(len(baseline)):
                    if i not in pair:
                         sample_design[i] = baseline[i] 
            
            print("sample_design:", sample_design)
            normalized_delta, normalized_params = prepare_data([sample_design], list_delta, 
                                                            data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])
            loss = compute_loss(config, problem_name, loss_function, pred_model, data_scaling_MM, 
                            normalized_params, normalized_delta, values)
            
            design_variables.append(sample_design)
            loss_per_sample.append(loss.detach().numpy().tolist())
        # print("Design variables:", design_variables)
        # print("Loss per sample:", loss_per_sample)
        
        all_design_variables.append(design_variables)
        all_losses_per_sample.append(loss_per_sample)
    np_all_losses_per_sample = np.array(all_losses_per_sample)
    
    # Number of samples 
    # size = 1
    # for dim in np.shape(np_all_losses_per_sample): size *= dim 
    # print("Number of samples:", size)
        
    # Display hotstart point i.e. the encountered point where the best results are:
    index_min = np.unravel_index(np.argmin(np_all_losses_per_sample, axis=None), np_all_losses_per_sample.shape)
    best_loss = np_all_losses_per_sample[index_min]
    best_design_variables = np.array(all_design_variables)[index_min]
    print("Loss of best encountered design is:", best_loss)
    print("Best encountered design is:",  best_design_variables)
    
    
    # Plot gradient map  
    from numpy import arange
    from numpy import meshgrid
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata
        
    for k, pair in enumerate(pairs):
        x = np.array([all_design_variables[k][i][pair[0]] for i in range(len(all_design_variables[k]))])
        y = np.array([all_design_variables[k][i][pair[1]] for i in range(len(all_design_variables[k]))])
        z = np.array(all_losses_per_sample[k])
        
        # Special regular grid conversion for continuous surface
        x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), n_samples), np.linspace(min(y), max(y), n_samples))
        z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')
        
        # Surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
                
        # Add best encountered point
        ax.scatter(best_design_variables[pair[0]], best_design_variables[pair[1]], best_loss, color='blue', s=100, label='Point')
        
        # Add baseline point
        # if baseline != None:
        #     ax.scatter(baseline[pair[0]], baseline[pair[1]], baseline_loss, color='red', s=100, label='Point')
        
        # Axes labeling and displaying
        ax.set_xlabel(name_design_vars[pair[0]])
        ax.set_ylabel(name_design_vars[pair[1]])
        ax.set_zlabel("Objective")
        plt.show()      
    
    
def grid_search_best(problem_name, name_config, n_samples):
    import itertools
    
    # Retrieve config
    config_lib = importlib.import_module("Models." + name_config + ".Config")
    config = config_lib.Config()
    name_design_vars = list(config.get_design_variables().keys())
        
    # Load the MLP and scalers
    pred_model, optimizer, data_scaling_params, data_scaling_MM = load_networks(config)
    loss_function = init_loss(problem_name) # Select corresponding loss
    
    # Init problem parameters
    if problem_name == "CalibrationMechanicalParams":
        list_s_a = [[0], [5], [10]]
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        simulation_values = init_problem(config, problem_name, list_s_a)
        values = [[act_disp for sublist in list_s_a for act_disp in sublist], simulation_values]
    elif problem_name == "DexterityObjective" or problem_name == "ContactForceObjective" or problem_name == "BendingAngleAndContactForceObjective":
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_s_a = [[10]] # Cable displacement used for evaluating the dexterity of the finger
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        values = [[act_disp for sublist in list_s_a for act_disp in sublist]]
                
    # Generate samples in the design data range
    n_vars = len(config.get_design_variables())
    n_sample_interval = max(2, int(np.exp(np.log(n_samples)/n_vars)))
    sampling_01 = list(np.linspace(0, 1, n_sample_interval))
    all_sampling01 = [sampling_01 for i in range(n_vars)]
    sampled_vars = [list(x) for x in np.array(np.meshgrid(*all_sampling01)).T.reshape(-1,len(all_sampling01))]
                            
    # Compute objective for sample
    all_design_variables = []
    all_losses_per_sample = []
    for sample_design_0_1 in sampled_vars:
        # Interpolate and retrieve real sampled design
        sample_design = config.interpolate_variables(sample_design_0_1, var_type = "design")

        normalized_delta, normalized_params = prepare_data([sample_design], list_delta, 
                                                        data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])
        loss = compute_loss(config, problem_name, loss_function, pred_model, data_scaling_MM, 
                        normalized_params, normalized_delta, values)
        
        all_design_variables.append(sample_design)
        all_losses_per_sample.append(loss.detach().numpy().tolist())
    
    np_all_losses_per_sample = np.array(all_losses_per_sample)
            
    # Display best point:
    index_min = np.unravel_index(np.argmin(np_all_losses_per_sample, axis=None), np_all_losses_per_sample.shape)
    best_loss = np_all_losses_per_sample[index_min]
    best_design_variables = np.array(all_design_variables)[index_min]
    print("Loss of best encountered design is:", best_loss)
    print("Best encountered design is:",  best_design_variables)
    
    # Display worst point:
    index_max = np.unravel_index(np.argmax(np_all_losses_per_sample, axis=None), np_all_losses_per_sample.shape)
    worst_loss = np_all_losses_per_sample[index_max]
    worst_design_variables = np.array(all_design_variables)[index_max]
    print("Loss of worst encountered design is:", worst_loss)
    print("Worst encountered design is:",  worst_design_variables)
    

def grid_pareto(problem_name, name_config, n_samples):
    import itertools
    
    # Retrieve config
    config_lib = importlib.import_module("Models." + name_config + ".Config")
    config = config_lib.Config()
    name_design_vars = list(config.get_design_variables().keys())
        
    # Load the MLP and scalers
    pred_model, optimizer, data_scaling_params, data_scaling_MM = load_networks(config)
    loss_function = init_loss(problem_name) # Select corresponding loss
    
    # Init problem parameters
    if problem_name == "BendingAngleAndContactForceObjective":
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_s_a = [[10]] # Cable displacement used for evaluating the dexterity of the finger
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        values = [[act_disp for sublist in list_s_a for act_disp in sublist]]
                
    # Generate samples in the design data range
    n_vars = len(config.get_design_variables())
    n_sample_interval = max(2, int(np.exp(np.log(n_samples)/n_vars)))
    sampling_01 = list(np.linspace(0, 1, n_sample_interval))
    all_sampling01 = [sampling_01 for i in range(n_vars)]
    sampled_vars = [list(x) for x in np.array(np.meshgrid(*all_sampling01)).T.reshape(-1,len(all_sampling01))]
                            
    # Compute objective for sample
    all_design_variables = []
    all_losses_per_sample = []
    for sample_design_0_1 in sampled_vars:
        # Interpolate and retrieve real sampled design
        sample_design = config.interpolate_variables(sample_design_0_1, var_type = "design")

        normalized_delta, normalized_params = prepare_data([sample_design], list_delta, 
                                                        data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])
        losses = compute_loss(config, problem_name, loss_function, pred_model, data_scaling_MM, 
                        normalized_params, normalized_delta, values, pareto_mode = True)
        
        all_design_variables.append(sample_design)
        all_losses_per_sample.append([loss.detach().numpy().tolist() for loss in losses])
    
    np_all_losses_per_sample = np.array(all_losses_per_sample)
    
    # Plot Pareto front evaluated on a grid
    import mplcursors
    x = [all_losses_per_sample[i][0] for i in range(len(all_losses_per_sample))]
    y = [all_losses_per_sample[i][1] for i in range(len(all_losses_per_sample))]
    plt.scatter(x, y)
    cursor = mplcursors.cursor(hover=True)
    
    cursor.connect('add', lambda sel: sel.annotation.set_text(all_design_variables[sel.target.index]))
    
    plt.xlabel("Dexterity")
    plt.ylabel("Strength")
    plt.show()

    
if __name__ == "__main__":
    main()