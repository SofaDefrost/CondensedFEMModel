""" This script implements methods for surrogate-assisted optimization of design parameters"""
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Apr 12 2023"

# System libs
import sys
import pathlib
import importlib

# Local libs
import numpy as np
import torch
import time
import copy
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
from Applications.TestQPGUI import ask_user_NN_properties
from Libraries.utils import init_network
MLPlearning_tools = importlib.import_module("Libraries.Learning.MLP.learning_tools")


class paramsToMMNetwork(torch.nn.Module):
    """Network to pass from params to MM. Composed of two networks:
    1. params -> (W0, dfree_0)
    2. (sa, W0, dfree_0) -> (W_t, dfree_t)
    """

    def __init__(self, config, n_constraint, data_scaling_params, data_scaling_MM, model_params, model_MM):
        """Initialization.

        Parameters:
        -----------
            n_constraint: int
                Number of constraint.
            model_params: torch.nn.Module
                Network to realise params -> (W0, dfree_0)
            model_MM: torch.nn.Module
                Network to realise (sa, W0, dfree_0) -> (W_t, dfree_t)
        """
        super(paramsToMMNetwork, self).__init__()
        self.config = config
        self.n_constraints = n_constraint
        self.data_scaling_params = data_scaling_params
        self.data_scaling_MM = data_scaling_MM
        self.model_params = model_params
        self.model_MM = model_MM
        

    def forward(self, params, delta):
        """Classical forward method of torch.nn.Module.

        Parameters:
        -----------
            params: tensor, requires_grad = True
                The input of the first network.
            delta: tensor, requires_grad = False
                Unpredicted input of the second network.

        Returns:
        --------
            The output of the network.
        """
        # Prediction from design param -> (W0, dfree_0)
        W0dfree0 = self.model_params(params)       
        
        # Rescale prediction to real values
        dfree_0_pred = W0dfree0[-self.n_constraints:] 
        W_pred = W0dfree0[:-self.n_constraints]
        W_0_pred = torch.zeros(size=(self.n_constraints, self.n_constraints))
        W_0_pred[torch.triu_indices(self.n_constraints, self.n_constraints)[0], torch.triu_indices(self.n_constraints, self.n_constraints)[1]] = W_pred
        saved_value = W_0_pred[torch.triu_indices(self.n_constraints, self.n_constraints, offset=1)[0],
                    torch.triu_indices(self.n_constraints, self.n_constraints, offset=1)[1]]
        W_0_pred.mT[torch.triu_indices(self.n_constraints, self.n_constraints, offset=1)[0],
        torch.triu_indices(self.n_constraints, self.n_constraints, offset=1)[1]] = saved_value
        
        if self.config.config_network["data_normalization"] == "Std":
            dfree_0_pred = dfree_0_pred * (torch.tensor(self.data_scaling_params[3][1])) + torch.tensor(self.data_scaling_params[2][1]) 
        elif self.config.config_network["data_normalization"] == "MinMax":  
            dfree_0_pred = dfree_0_pred * (torch.tensor(self.data_scaling_params[3][1])  - torch.tensor(self.data_scaling_params[2][1])) + torch.tensor(self.data_scaling_params[2][1])
        
        if self.config.config_network["data_normalization"] == "Std":
            W_0_pred = W_0_pred * (torch.tensor(self.data_scaling_params[3][0])) + torch.tensor(self.data_scaling_params[2][0])
        elif self.config.config_network["data_normalization"] == "MinMax":
            W_0_pred = W_0_pred * (torch.tensor(self.data_scaling_params[3][0]) - torch.tensor(self.data_scaling_params[2][0])) + torch.tensor(self.data_scaling_params[2][0])
                  
        W_0_pred = W_0_pred.reshape(self.n_constraints, self.n_constraints)
        
        #print("W_0_pred:", W_0_pred)
        #print("dfree_0_pred:", dfree_0_pred)
        
        # Standardize for second network (W0, dfree_0, delta_t) -> (W_t, dfree_t)          
        X_MM = [W_0_pred, dfree_0_pred, torch.tensor(self.n_constraints * [0])]
        Y_MM = [W_0_pred, dfree_0_pred]
        if self.config.config_network["data_normalization"] == "Std":
            X_MM, _ = MLPlearning_tools.create_data_std(X_MM, Y_MM, self.data_scaling_MM[0], self.data_scaling_MM[1], self.data_scaling_MM[2], self.data_scaling_MM[3], design_to_MM=False, torch_mode = True)
        elif self.config.config_network["data_normalization"] == "MinMax":
            X_MM, _ = MLPlearning_tools.create_data_minmax(X_MM, Y_MM, self.data_scaling_MM[0], self.data_scaling_MM[1], self.data_scaling_MM[2], self.data_scaling_MM[3], design_to_MM=False, torch_mode = True)
        else:
            print("[ERROR] >> normalization_method should be in [Std, MinMax]")
            exit(1)
        
        # Get standardize W_0 and Dfree_0
        W0dfree0 = X_MM[:-self.n_constraints]
        #print("W0dfree0:", W0dfree0)
        
        # Predict (W_t, dfree_t)
        inputMM = torch.cat([W0dfree0, delta], axis=0).float()
        #print("inputMM:", inputMM)
        Wtdfreet = self.model_MM(inputMM)
        #print("Wtdfreet:", Wtdfreet)
        return self.model_MM(inputMM)

def load_networks(config):
    # Load network for params -> (W0, dfree_0)
    print("[INFO] >> Find the config for the design_to_MM netowk.")
    config.config_network = ask_user_NN_properties(config.model_name, "MLP")

    # Random init of dynamical unused parameters of the learnign process
    config.config_network["learning_rate"] = 0.0001
    config.config_network["dropout_probability"] = 0

    # Retrieve learned network
    model_params, _, _, _, _, n_constraint, data_scaling_params, _, _, _, best_model_link_params, _ = init_network("MLP", config, design_to_MM=True)
    if pathlib.Path.exists(best_model_link_params):
        checkpoint = torch.load(best_model_link_params)
        model_params.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("[ERROR] >> No best model for the design params network.")
        exit(1)

    # Load network for (W0, dfree_0) -> (Wt, dfree_t)
    print("[INFO] >> Find the config for the (W0, dfree_0) -> (Wt, dfree_t) netowk.")
    config.config_network = ask_user_NN_properties(config.model_name, "MLP")

    # Random init of dynamical unused parameters of the learnign process
    config.config_network["learning_rate"] = 0.0001
    config.config_network["dropout_probability"] = 0

    # Retrieve learned network
    model_MM, _, _, _, _, _, data_scaling_MM, _, _, _, best_model_link_MM, _ = init_network("MLP", config, design_to_MM=False)
    if pathlib.Path.exists(best_model_link_MM):
        checkpoint = torch.load(best_model_link_MM)
        model_MM.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("[ERROR] >> No best model for the (W0, dfree_0) -> (Wt, dfree_t) network.")
        exit(1)

    model = paramsToMMNetwork(config, n_constraint, data_scaling_params, data_scaling_MM, model_params, model_MM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Optimizer

    return model, optimizer, data_scaling_params, data_scaling_MM



def prepare_data(data_design, batch_delta, data_scaling_params, data_scaling_MM, n_constraints, normalization_method):
    """Normalize a batch of data.
    ----------
    Parameters
    ----------
    data_design: list of floats
        Values for each design varaibles.
    batch_delta: list of list of floats
        Batch of actuation/effector states.

    data_scaling_params: list of lists of floats
        Scaling parameters for design parameters
    data_scaling_MM; list of lists of floats
        Scaling parameters for mechanical matrices
    
    n_constraints: int
        Number of constraints in the problem

    normalization_method: string in {None, MinMax Std}
    ----------
    Outputs
    ----------
    normalized_data_design:

    normalized_data_MM:

    """

    W_0 = np.zeros((n_constraints, n_constraints))
    dfree_0 = np.zeros(n_constraints)

    X_MM_batch = [torch.tensor([W_0 for i in range(len(batch_delta))]), torch.tensor([dfree_0 for i in range(len(batch_delta))]),
    torch.tensor([batch_delta[i] for i in range(len(batch_delta))])]
    Y_MM_batch = [torch.tensor([W_0 for i in range(len(batch_delta))]), torch.tensor([dfree_0 for i in range(len(batch_delta))])]
    
    if normalization_method == "Std":
        X_MM, _ = MLPlearning_tools.create_data_std(X_MM_batch, Y_MM_batch, data_scaling_MM[0], data_scaling_MM[1], data_scaling_MM[2], data_scaling_MM[3], design_to_MM=False)
    elif normalization_method == "MinMax":
        X_MM, _ = MLPlearning_tools.create_data_minmax(X_MM_batch, Y_MM_batch, data_scaling_MM[0], data_scaling_MM[1], data_scaling_MM[2], data_scaling_MM[3], design_to_MM=False)
    else:
        print("[ERROR] >> normalization_method should be in [Std, MinMax]")
        exit(1)

    #For design params
    X_params_batch = [torch.tensor([data_design[i] for i in range(len(data_design))])]
    Y_params_batch = [torch.tensor([W_0 for i in range(len(batch_delta))]), torch.tensor([dfree_0 for i in range(len(batch_delta))])]
    # print("X_params_batch:", X_params_batch)
    
    if normalization_method == "Std":
        X_params, _ = MLPlearning_tools.create_data_std(X_params_batch, Y_params_batch, data_scaling_params[0], data_scaling_params[1],
                                                    data_scaling_params[2], data_scaling_params[3], design_to_MM=True, in_DO_loop = True)
    elif normalization_method == "MinMax":
        X_params, _ = MLPlearning_tools.create_data_minmax(X_params_batch, Y_params_batch, data_scaling_params[0], data_scaling_params[1],
                                                       data_scaling_params[2], data_scaling_params[3], design_to_MM=True)
    else:
        print("[ERROR] >> normalization_method should be in [Std, MinMax]")
        exit(1)
    # print("X_params:", X_params)
        
    return X_MM[:, -n_constraints:], X_params[0]


def design_optimization(config, n_step = 1500, learning_rate = 1e-1, save_results = True): 

    # Load condensed FEM model networks
    pred_model, _, data_scaling_params, data_scaling_MM = load_networks(config)

    # Choose optimization case
    problem_name = pick_optimization_problem(config)
    loss_function = init_loss(problem_name) # Select corresponding loss

    # Init data     
    init_learning_rate = learning_rate
    init_design_params = [[param[0] for param in config.get_design_variables().values()]]
    names_design_params = [param for param in config.get_design_variables().keys()]
    print("Initialization of design parameters:", init_design_params)
    print("Names of design parameters:", names_design_params)
   
    # Init problem
    if problem_name == "CalibrationMechanicalParams":
        # Compute ground truth data used for calibration
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_s_a = [[0], [5], [10]] # [[0], [2.5], [5], [7.5], [10]]
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        simulation_values = init_problem(config, problem_name, list_s_a)
        values = [[act_disp for sublist in list_s_a for act_disp in sublist], simulation_values]
        
    elif problem_name == "CalibrationMechanicalParamsForce":
        # Compute ground truth data used for calibration
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_lambda_a = [[100]] # [[0], [100], [200]] 
        list_delta = [list_lambda_a[i]+[0]*n_const_effectors for i in range(len(list_lambda_a))]  
        simulation_values = init_problem(config, problem_name, list_lambda_a)
        values = [[act_disp for sublist in list_lambda_a for act_disp in sublist], simulation_values]
        
    elif problem_name == "DexterityObjective" or problem_name == "ContactForceObjective" or problem_name == "BendingAngleAndContactForceObjective":
        n_const_effectors = pred_model.n_constraints - len(config.get_actuators_variables())
        list_s_a = [[10]] # Cable displacement used for evaluating the dexterity of the finger
        list_delta = [list_s_a[i]+[0]*n_const_effectors for i in range(len(list_s_a))]  
        values = [[act_disp for sublist in list_s_a for act_disp in sublist]]

        
    else:
        pass

    # Normalized data
    normalized_delta, normalized_params = prepare_data(init_design_params, list_delta, data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])   

    # Normalize design variable bounds
    bmin_design_params = [[param[1] for param in config.get_design_variables().values()]]
    _, normalized_bmin_design_params = prepare_data(bmin_design_params, list_delta, data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])
    bmax_design_params = [[param[2] for param in config.get_design_variables().values()]]
    _, normalized_bmax_design_params = prepare_data(bmax_design_params, list_delta, data_scaling_params, data_scaling_MM, pred_model.n_constraints, config.config_network["data_normalization"])
        
    # Add learning rate scheduler
    USE_SCHEDULER = False
    if USE_SCHEDULER:
        factor = 0.1
        patience = 50
        min_lr = 1e-3
        # Special counters
        best_loss = 1000000
        no_improvement_count = 0

    # Main optimization loop for single objective optimization
    if save_results:
        params_per_iter = []
        loss_per_iter = []
    input_params = normalized_params
    input_params.requires_grad = True    
    for _ in range(n_step):
        input_params.grad = None
        loss = compute_loss(config, problem_name, loss_function, pred_model, data_scaling_MM, 
                            input_params, normalized_delta, values, printlog = (_ == n_step -1))
        if save_results:
            params_per_iter.append(unscale_design_params(config, input_params.detach().numpy(), data_scaling_params).tolist()[0])
            loss_per_iter.append(loss.detach().numpy().tolist())
        loss.backward()
        # TODO: add a scheduling for reducing the learnign rate
        print("Gradient:", input_params.grad)
        print("Input params before applying gradient:",  input_params.data)
        # print("learning rate:", learning_rate)
        input_params.data = torch.min(torch.max(input_params.data - learning_rate* input_params.grad, normalized_bmin_design_params), normalized_bmax_design_params)
        print("Input params after applying gradient:",  input_params.data)
            
        # Reduce learning rate if no more improvement has been made for 10 iterations
        if USE_SCHEDULER:
            print("Current learning rate:", learning_rate)
            if loss < best_loss:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                learning_rate = max(learning_rate * factor, min_lr)
                no_improvement_count = 0  


    #Unscaled the params                    
    params = unscale_design_params(config, input_params.detach().numpy(), data_scaling_params)
    print("[INFO] >> The best params are: ", params)

    # Generate a plot of the results
    if save_results:
        init_params_string = ""
        for i in range(len(init_design_params[0])):
            init_params_string += "_" + str(init_design_params[0][i]) + "p" + str(i)
        name_DO_results = problem_name + "_" + str(n_step) + "it_" + str(init_learning_rate) + "lr" + init_params_string
        np.save("Results/DesignOptimization/OptimizationHistories/" + name_DO_results + "_param_names.npy", np.array(names_design_params))
        np.save("Results/DesignOptimization/OptimizationHistories/" + name_DO_results + "_loss_hist.npy", np.array(loss_per_iter))
        np.save("Results/DesignOptimization/OptimizationHistories/" + name_DO_results + "_params_hist.npy", np.array(params_per_iter))
        
        #print("params_per_iter:", params_per_iter)
        #print("loss_per_iter:", loss_per_iter)
        
        # Plot objective history
        plot_obj_history(loss_per_iter)
        #plt.savefig("")
        
        # Plot parameters convergence history
        plot_params_history(names_design_params, params_per_iter)
        #plt.savefig("")


def unscale_design_params(config, normalized_params, data_scaling_params):
    """
    Unscale a given set of normalize design parameters
    """
    if config.config_network["data_normalization"] == "Std":
        params = normalized_params * (data_scaling_params[1]) + data_scaling_params[0]
    elif config.config_network["data_normalization"] == "MinMax":
        params = normalized_params * (data_scaling_params[1]  - data_scaling_params[0]) + data_scaling_params[0]
    return params

def plot_obj_history(loss_per_iter):
    x = list(range(len(loss_per_iter)))
    y = loss_per_iter
    plt.scatter(x, y,  alpha=0.5)
    plt.show()
    
def plot_params_history(names_design_params, params_per_iter):
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
    host.legend()
    plt.draw()
    plt.show()
     
    
    

### Methods implementing different optimization problems features
def pick_optimization_problem(config):
    """
    User selection of the optimization problem to solve.
    """
    # List of available optimization problems
    if config.model_name == "FingerElasticityParams":
        problems = ["CalibrationMechanicalParams", "CalibrationMechanicalParamsForce"]
    elif config.model_name == "FingerDesign" or config.model_name == "FingerDesignLength":
        problems = ["DexterityObjective", "ContactForceObjective", "BendingAngleAndContactForceObjective"]

    # Loop for picking one
    print("Optimization problem available for ", config.model_name, ":")
    for i, problem in enumerate(problems):
        print(">> Optimization problem n°  ", i, ":", problem)

    user_input_model = int(input("What optimization problem do you want to use? "))
    while user_input_model<0 or user_input_model>i:
        print("Please answer a number in:", [0, i])
        user_input_model = int(input("What optimization problem do you want to use? "))

    return problems[user_input_model]

def init_problem(config, problem_name, inputs):
    """
    Function used for initialization of special parameters for a given problem.
    ----------
    Parameters
    ----------
    config: Config
        Class describing the model
    problem_name: string
        Name of the problem to solve.
    inputs: list of whatever
        List of specific inputs for the given problem.
    ----------
    Outputs
    ----------
    outputs: list of whatever
        List of specific outputs for the given problem.
    """

    if problem_name == "CalibrationMechanicalParams":
        ### Get actuation states
        sampled_s_a = inputs 

        ### Let's generate the ground truth data
        config.PoissonRation = 0.47 # 0.47
        config.YoungsModulus = 3000 # 3000

        ### Simulation using SOFA
        # On a real world application, we would just provide physical measures instead
        import Sofa
        import SofaRuntime
        SofaRuntime.importPlugin("SofaPython3")
        sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
        scene_lib = importlib.import_module("Models." + config.scene_name + "." + config.model_name) 

        n_eq_dt = config.get_n_eq_dt()
        n_dt = config.get_n_dt()
        post_sim_n_eq_dt = config.get_post_sim_n_eq_dt() 

        # Compute effector states for sampled actuation displacement states
        outputs = []
        for constraints_vars in sampled_s_a:
            root = Sofa.Core.Node("root")
            scene_lib.createScene(root, config)
            Sofa.Simulation.init(root)

            # Reach equilibrium
            null_action = (len(config.get_actuators_variables()) + len(config.get_contacts_variables())) * [0]
            root.Controller.apply_actions(null_action)
            for step in range(n_eq_dt):
                Sofa.Simulation.animate(root, root.dt.value)
                time.sleep(root.dt.value)
                
            dfree0 = copy.deepcopy(root.Controller.get_dfree())
            W0 = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())

            # Apply actuation / contact displacement
            for step in range(n_dt + post_sim_n_eq_dt):
                constraints_vars_step = [min((step + 1) * v/(n_dt), v) for v in constraints_vars] # Apply gradually action
                root.Controller.apply_actions(constraints_vars_step)
                Sofa.Simulation.animate(root, root.dt.value)
                time.sleep(root.dt.value)

            
            print(">> Simulation starts ...")
            dfree = copy.deepcopy(root.Controller.get_dfree())
            W = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
              
    
            # Retrieve distance effectors / goals
            effectors_state = copy.deepcopy(root.Controller.get_effectors_state_calibration())
            print("Effectors_state for calibration:", effectors_state)
            outputs.append(effectors_state)
            print(">> Simulation done")
            
            # Reset simulation
            Sofa.Simulation.reset(root)
        
        print("Target displacements to reach:", outputs)
        
                
    elif problem_name == "CalibrationMechanicalParamsForce":
        ### Get applied forces
        sampled_s_a = inputs 
        config.scene_config["is_force"] = True

        ### Let's generate the ground truth data
        config.PoissonRation = 0.47 # 0.47
        config.YoungsModulus = 3000 # 3000

        ### Simulation using SOFA
        # On a real world application, we would just provide physical measures instead
        import Sofa
        import SofaRuntime
        SofaRuntime.importPlugin("SofaPython3")
        sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
        scene_lib = importlib.import_module("Models." + config.scene_name + "." + config.model_name) 

        n_eq_dt = config.get_n_eq_dt()
        n_dt = config.get_n_dt()
        post_sim_n_eq_dt = config.get_post_sim_n_eq_dt() 

        # Compute effector states for sampled actuation displacement states
        outputs = []
        for constraints_vars in sampled_s_a:
            root = Sofa.Core.Node("root")
            scene_lib.createScene(root, config)
            Sofa.Simulation.init(root)

            # Reach equilibrium
            null_action = (len(config.get_actuators_variables()) + len(config.get_contacts_variables())) * [0]
            root.Controller.apply_actions(null_action)
            for step in range(n_eq_dt):
                Sofa.Simulation.animate(root, root.dt.value)
                time.sleep(root.dt.value)
                
            dfree0 = copy.deepcopy(root.Controller.get_dfree())
            W0 = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())

            # Apply actuation / contact displacement
            for step in range(n_dt + post_sim_n_eq_dt):
                constraints_vars_step = [min((step + 1) * v/(n_dt), v) for v in constraints_vars] # Apply gradually action
                root.Controller.apply_actions(constraints_vars_step)
                Sofa.Simulation.animate(root, root.dt.value)
                time.sleep(root.dt.value)

            
            print(">> Simulation starts ...")
            dfree = copy.deepcopy(root.Controller.get_dfree())
            W = copy.deepcopy(root.Controller.get_compliance_matrice_in_constraint_space())
              
            # Retrieve distance effectors / goals
            effectors_state = copy.deepcopy(root.Controller.get_effectors_state_calibration())
            print("Effectors_state for calibration:", effectors_state)
            outputs.append(effectors_state)
            print(">> Simulation done")
            
            # Reset simulation
            Sofa.Simulation.reset(root)
        
        print("Target displacements to reach:", outputs)
    

    return outputs


def init_loss(problem_name):
    """
    Initialize loss function corresponding to given problem.
    """
    
    if problem_name == "CalibrationMechanicalParams" or problem_name == "CalibrationMechanicalParamsForce":
        loss_function = torch.nn.L1Loss()
    if problem_name == "DexterityObjective" or problem_name == "ContactForceObjective" or problem_name == "BendingAngleAndContactForceObjective":
        loss_function = torch.nn.L1Loss()
    return loss_function

def compute_loss(config, problem_name, loss_function, pred_model, data_scaling_MM, normalized_design, normalized_delta, values, pareto_mode = False, printlog = False):
    """
    Compute loss value.
    ----------
    Parameters
    ----------
    config: Config
    problem_name: string
        Name of the problem to solve.
    loss_function: torch.nn.func
        Loss functions.
    normalized_delta: list fo floats
        Normalized constraints displacement states for calling the prediction model.
    pred_model: torch.nn.Model
        Learned model linking design parameters and actuation state to mechancial matrices.
    data_scaling_MM: list of list of floats
        Scaling factors for data normalziation of mechanical matrices.
    normalized_design: list of float
        Normalized design variable values.
    values: list of list of float
        Values of elements for computing the loss function.
        Nature of provided values depends on the type of problem.
    pareto_mode; bool
        Indicates if we are in Pareto mode.
    printlog: boolean
        Indicate if we pritn special logs or not.
        
    ----------
    Outputs
    ----------
    loss: float
        Computed loss
    """
    
    def get_and_rescale_mechanical_matrices(config, pred_model, dfree, W):
        # Rescale dfree
        if config.config_network["data_normalization"] == "Std":
            dfree = dfree * (torch.tensor(data_scaling_MM[3][1])) + torch.tensor(data_scaling_MM[2][1]) # Rescale dfree
        elif config.config_network["data_normalization"] == "MinMax":
            dfree = dfree * (torch.tensor(data_scaling_MM[3][1])  - torch.tensor(data_scaling_MM[2][1])) + torch.tensor(data_scaling_MM[2][1]) # Rescale dfree
    
        # Rescale W
        W_pred = predicted_values[:-pred_model.n_constraints]
        W = torch.zeros(size=(pred_model.n_constraints, pred_model.n_constraints))
        W[torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints)[0], torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints)[1]] = W_pred
        saved_value = W[torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints, offset=1)[0],
                    torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints, offset=1)[1]]
        W.mT[torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints, offset=1)[0],
        torch.triu_indices(pred_model.n_constraints, pred_model.n_constraints, offset=1)[1]] = saved_value
        if config.config_network["data_normalization"] == "Std":
            W = W * (torch.tensor(data_scaling_MM[3][0])) + torch.tensor(data_scaling_MM[2][0])
        elif config.config_network["data_normalization"] == "MinMax":
            W = W * (torch.tensor(data_scaling_MM[3][0]) - torch.tensor(data_scaling_MM[2][0])) + torch.tensor(data_scaling_MM[2][0]) 
        
        return dfree, W

    #####################################
    # Calibration Mechanical parameters #
    #####################################
    if problem_name == "CalibrationMechanicalParams":
        
        # Loss is difference between effector position displacements of the Finger for several actuation displacement states.
        s_a = values[0] # actuation displacements state encountered during calibration
        target_delta_e = values[1] # measurer effector displacements
        loss = 0
        
        for i, norm_delta in enumerate(normalized_delta):
            predicted_values = pred_model(normalized_design[0], norm_delta)
                         
            # # Get and rescale dfree and W            
            dfree = predicted_values[-pred_model.n_constraints:]
            W = predicted_values[:-pred_model.n_constraints]
            dfree, W = get_and_rescale_mechanical_matrices(config, pred_model, dfree, W)
            
            
            # Compute delta_e from predicted_values 
            # We have delta_e = Wea lambda_a + delta_e^free
            #         delta_a = Waa lambda_a +  delta_a^free
            # Giving lambda_a = Waa^(-1)(delta_a - delta_a^free)
            # Then delta_e = Wea Waa-1 (delta_a - delta_a^free) + delta_e^free
                              
            n_act_constraint = len(config.get_actuators_variables())
            
            # Retrieve mechanical matrices
            dfree_a = dfree[:n_act_constraint]
            
            init_effector_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1)) 
            goal_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1))
            dfree_e = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
                
            Waa = W[:n_act_constraint, :n_act_constraint]
            Wea = W[n_act_constraint:, :n_act_constraint]     
                        
            # Compute delta_e
            Ds_a = (torch.tensor(s_a[i]) - dfree_a)[:, None]
            delta_e = torch.matmul(Wea, torch.matmul(torch.inverse(Waa), Ds_a)) + dfree_e 
    
            t_delta_e = torch.tensor(np.expand_dims(target_delta_e[i], axis=1)) + torch.tensor(np.expand_dims([0, 10, 0], axis=1)) # init_effector_pos - goal_pos # effector_disp + effector_init_pos - goal_pos
            
            # Compute loss
            loss += loss_function(delta_e,  t_delta_e)
            if printlog:
                print("Measure delta_e:", delta_e)
                print("Calibrated model delta_e:", t_delta_e)
                        
        print("Total loss:", loss)
        
        
    ################################################
    # Calibration Mechanical parameters with force #
    ################################################
    if problem_name == "CalibrationMechanicalParamsForce":
        # Calibration of mechanical params, but now using applied force on cables as ground truth.
        
        # Loss is difference between effector position displacements of the Finger for several actuation forces applied on the cable.
        lambda_a = values[0] # actuation forces encountered during calibration
        target_delta_e = values[1] # measure effector displacements
        loss = 0
        
        # Incremental computation of delta_a = Waa lambda_a +  delta_a^free
        # We try to find delta_a so that lambda_a = Waa-1 [delta_a - delta_a^free]      
        # TODO: Optimize delta_a for minimizing pred_lambda_a - lambda_a 
              # Deducing delta_a and then mechancial matrices ...
        
        for i, norm_delta in enumerate(lambda_a):
            predicted_values = pred_model(normalized_design[0], norm_delta)
                         
            # # Get and rescale dfree and W            
            dfree = predicted_values[-pred_model.n_constraints:]
            W = predicted_values[:-pred_model.n_constraints]
            dfree, W = get_and_rescale_mechanical_matrices(config, pred_model, dfree, W)
            
            # Compute delta_e from predicted_values 
            # We have delta_e = Wea lambda_a + delta_e^free                              
            n_act_constraint = len(config.get_actuators_variables())
            
            # Retrieve mechanical matrices            
            init_effector_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1)) 
            goal_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1))
            dfree_e = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
                
            Wea = W[n_act_constraint:, :n_act_constraint]     
                        
            # Compute delta_e
            delta_e = torch.matmul(Wea, torch.tensor(lambda_a[i])) + dfree_e 
            
            # Compute loss
            t_delta_e = torch.tensor(np.expand_dims(target_delta_e[i], axis=1)) + torch.tensor(np.expand_dims([0, 10, 0], axis=1)) # init_effector_pos - goal_pos # effector_disp + effector_init_pos - goal_pos
            loss += loss_function(delta_e,  t_delta_e)
            if printlog:
                print("Measure delta_e:", delta_e)
                print("Calibrated model delta_e:", t_delta_e)
                        
        print("Total loss:", loss)
    
    
        
    ##############################################
    # Bending angle for given cable dispalcement #
    ##############################################
    elif problem_name == "DexterityObjective":
        
        # Loss is computed by comparing reached bending angle for a fixed actuation displacement state with an unreachable angle.
        s_a = values[0] # Fixed actuation displacements state 
        
        for i, norm_delta in enumerate(normalized_delta):
            predicted_values = pred_model(normalized_design[0], norm_delta)
                         
            ### Get and rescale dfree and W            
            dfree = predicted_values[-pred_model.n_constraints:]
            W = predicted_values[:-pred_model.n_constraints]
            dfree, W = get_and_rescale_mechanical_matrices(config, pred_model, dfree, W)
            
            ### Compute the angle difference of the effector position between rest and actuated state in the Z_plane
            
            ## First we compute delta_e from predicted_values 
            # We have delta_e = Wea lambda_a + delta_e^free
            #         delta_a = Waa lambda_a +  delta_a^free
            # Giving lambda_a = Waa^(-1)(delta_a - delta_a^free)
            # Then delta_e = Wea Waa-1 (delta_a - delta_a^free) + delta_e^free             
            n_act_constraint = len(config.get_actuators_variables())
            
            # Retrieve mechanical matrices
            dfree_a = dfree[:n_act_constraint]
            
            init_effector_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1)) 
            goal_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1))
            dfree_e = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
                
            Waa = W[:n_act_constraint, :n_act_constraint]
            Wea = W[n_act_constraint:, :n_act_constraint]     
                        
            # Compute delta_e
            Ds_a = (torch.tensor(s_a[i]) - dfree_a)[:, None]
            delta_e = torch.matmul(Wea, torch.matmul(torch.inverse(Waa), Ds_a)) + dfree_e 
    
            ## Secondly, we compute the projected angle difference 
            # print("init_effector_pos:", init_effector_pos)
            # print("delta_e:", delta_e)
            effector_pos = init_effector_pos + delta_e 
            print("effector_pos:", effector_pos)
            
            MAX_ANGLE = torch.tensor(1.57) #90°
            angle = torch.acos(torch.div(torch.abs(effector_pos[2]), torch.norm(init_effector_pos)))
            print("angle: ", angle)
            loss = loss_function(angle, MAX_ANGLE)
            print("loss:", loss)
            
            
    ##############################################
    # Contact force for given cable displacement #
    ##############################################
    elif problem_name == "ContactForceObjective":
        
        # Loss is computed by comparing contacc force generated on the tip effector for a fixed actuation displacement state 
        # Under the asusmption that the object is not moving and in contact with the tip of the finger 
        s_a = values[0] # Fixed actuation displacements state 
        
        for i, norm_delta in enumerate(normalized_delta):
            predicted_values = pred_model(normalized_design[0], norm_delta)
                         
            ### Get and rescale dfree and W            
            dfree = predicted_values[-pred_model.n_constraints:]
            W = predicted_values[:-pred_model.n_constraints]
            dfree, W = get_and_rescale_mechanical_matrices(config, pred_model, dfree, W)
                        
            ## First we compute lambda_c from predicted_values 
            # We have delta_c = Wcc lambda_c + Wca lambda_a + delta_c^free = 0 under the assumption of object in contact
            #         delta_a = Waa lambda_a + Wac lambda_c + delta_a^free   
            # Giving lambda_a = Waa-1 [delta_a - delta_a^free - Wac lambda_c]
            # Then 0 = Wcc lambda_c + Wca Waa-1 [delta_a - delta_a^free - Wac lambda_c] + delta_c^free
            # [Wca Waa-1 Wac - Wcc] lambda_c =  Wca Waa-1 [delta_a - delta_a^free] + delta_c^free
            # lambda_c = A-1 [Wca Waa-1 Ds_a + delta_c^free]
                                   
            n_act_constraint = len(config.get_actuators_variables())
            
            # Retrieve mechanical matrices
            init_effector_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1)) 
            goal_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1))
            dfree_a = dfree[:n_act_constraint]
            dfree_c = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
             
            Waa = W[:n_act_constraint, :n_act_constraint]
            Wac = W[:n_act_constraint, n_act_constraint:]
            Wca = W[n_act_constraint:, :n_act_constraint]
            Wcc = W[n_act_constraint:, n_act_constraint:]     
                        
            # Compute lambda_c
            Ds_a = (torch.tensor(s_a[i]) - dfree_a)[:, None]
            A = torch.matmul(Wca, torch.matmul(torch.inverse(Waa), Wac)) - Wcc
            lambda_c = torch.matmul(torch.inverse(A), torch.matmul(Wca, torch.matmul(torch.inverse(Waa), Ds_a)) + dfree_c)
            print("Generated contact force:", lambda_c[1])
            
            # Compute loss as the difference between the generated force along y-axis and an unreachable force.       
            MAX_FORCE = torch.tensor(10000) 
            #print("lambda_c[1] / MAX_FORCE:", torch.abs(lambda_c[1]) / MAX_FORCE)
            
            loss = loss_function(torch.abs(lambda_c[1]) / MAX_FORCE, torch.tensor(1))
            print("loss:", loss)
            
            
    ################################################################
    # Bending Angle and Contact force for given cable displacement #
    ################################################################
    elif problem_name == "BendingAngleAndContactForceObjective":
        
        # Loss is a combination of both loss for bending angle and contact force generated on the tip of the finger
        # Under fixed cable displacement
        
        s_a = values[0] # Fixed actuation displacements state 
        
        for i, norm_delta in enumerate(normalized_delta):
            predicted_values = pred_model(normalized_design[0], norm_delta)
                         
            ### Get and rescale dfree and W            
            dfree = predicted_values[-pred_model.n_constraints:]
            W = predicted_values[:-pred_model.n_constraints]
            dfree, W = get_and_rescale_mechanical_matrices(config, pred_model, dfree, W)
            n_act_constraint = len(config.get_actuators_variables())
                        
            ## First we compute lambda_c from predicted_values 
            # We have delta_c = Wcc lambda_c + Wca lambda_a + delta_c^free = 0 under the assumption of object in contact
            #         delta_a = Waa lambda_a + Wac lambda_c + delta_a^free   
            # Giving lambda_a = Waa-1 [delta_a - delta_a^free - Wac lambda_c]
            # Then 0 = Wcc lambda_c + Wca Waa-1 [delta_a - delta_a^free - Wac lambda_c] + delta_c^free
            # [Wca Waa-1 Wac - Wcc] lambda_c =  Wca Waa-1 [delta_a - delta_a^free] + delta_c^free
            # lambda_c = A-1 [Wca Waa-1 Ds_a + delta_c^free]
                                   
            
            # Retrieve mechanical matrices
            # Here the effector is viewed wether as an effector or a contact point depending of the considered loss
            init_effector_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1)) 
            goal_pos = torch.tensor(np.expand_dims(config.scene_config["goalPos"], axis=1))
            dfree_a = dfree[:n_act_constraint]
            dfree_e = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
            dfree_c = dfree[n_act_constraint:][:, None] + init_effector_pos - goal_pos # (Init_pos + Relative_Disp) - Goal_pos
             
            Waa = W[:n_act_constraint, :n_act_constraint]
            Wea = W[n_act_constraint:, :n_act_constraint] 
            Wac = W[:n_act_constraint, n_act_constraint:]
            Wca = W[n_act_constraint:, :n_act_constraint]
            Wcc = W[n_act_constraint:, n_act_constraint:]     
                        
            Ds_a = (torch.tensor(s_a[i]) - dfree_a)[:, None]
            
            # Compute bending angle loss
            Ds_a = (torch.tensor(s_a[i]) - dfree_a)[:, None]
            delta_e = torch.matmul(Wea, torch.matmul(torch.inverse(Waa), Ds_a)) + dfree_e 
            effector_pos = init_effector_pos + delta_e 
            MAX_ANGLE = torch.tensor(1.57) #90°
            angle = torch.acos(torch.div(torch.abs(effector_pos[2]), torch.norm(init_effector_pos)))
            loss_bending_angle = loss_function(angle, MAX_ANGLE)
            
            # Compute contact force loss
            A = torch.matmul(Wca, torch.matmul(torch.inverse(Waa), Wac)) - Wcc
            lambda_c = torch.matmul(torch.inverse(A), torch.matmul(Wca, torch.matmul(torch.inverse(Waa), Ds_a)) + dfree_c)    
            MAX_FORCE = torch.tensor(10000)             
            loss_contact_force = loss_function(torch.abs(lambda_c[1]) / MAX_FORCE, torch.tensor(1))
    
            # Final Loss 
            n_loss_bending_angle = (loss_bending_angle - torch.tensor(0.978596)) / torch.tensor(1.145478 - 0.978596) 
            n_loss_contact_force = (loss_contact_force- torch.tensor(0.525234)) / torch.tensor(0.999853 - 0.525234)
              
            gamma_1 = torch.tensor(1)
            gamma_2 = torch.tensor(0.8) 
            loss =  gamma_1 * n_loss_bending_angle +  gamma_2 * n_loss_contact_force 
            print("Normalized weighted loss_bending_angle:", gamma_1 * n_loss_bending_angle)
            print("Normalized weighted loss_contact_force:", gamma_2 * n_loss_contact_force)

            if pareto_mode:
                loss = [n_loss_bending_angle, n_loss_contact_force]
                        
    return loss