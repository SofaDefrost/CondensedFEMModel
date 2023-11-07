""" This script aims at analysing acquired data in order for helping to choose the best learning strategy."""
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
import pandas as pd
import tabloo
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../") 
from Libraries.database import query_simulation_data
from Libraries.utils import ask_user_NN_properties, ask_user_sampling_properties

# Available data analysis modes 
list_modes = ["EXPLORE_DATA", # Explore data correlations in a table format
              "EXPLORE_DATA_RANGE", # Explore data distribution
              "EXPLORE_DATA_CORRELATION", # Explore data correlation in a plot format
              "EXPLORE_DATA_WORKSPACE_TRAINING", # Plot data workspace for a training set
                "EXPLORE_DATA_WORKSPACE_TEST", # Plot data workspace for a test set
                "EXPLORE_LEARNING_OVERFITTING",  # Compute metrics for assessing prediction ability on the train set
                "EXPLORE_LEARNING_VALIDATION", # Compute metrics for assessing prediction ability on the test set
                "EXPLORE_LEARNING_OVERFITTING_DESIGN_PIPELINE", # Compute metrics for assessing prediction ability on the train set with design NN
                "EXPLORE_LEARNING_VALIDATION_DESIGN_PIPELINE", # Compute metrics for assessing prediction ability on the test set with design NN
                
                
                
                ]


def data_analysis(config, design_to_MM):
    """
    Analyse acquired data.
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    design_to_MM: Bool
        Specify if the data have to be in the format use for learnign design parameters
    """
    WITH_DESIGN_PARAMS = False
    
    # Choose mode for data analysis
    analysis_mode = ask_user_analysis_mode(list_modes)

    # Get sampling data
    data_params = ask_user_sampling_properties(config.config_network)

    is_train = True
    if analysis_mode in ["EXPLORE_LEARNING_VALIDATION", "EXPLORE_LEARNING_VALIDATION_DESIGN_PIPELINE", "EXPLORE_DATA_WORKSPACE_TEST"]:
        is_train = False
    binary_data = list(query_simulation_data(config.model_name, data_params["id"], is_train))

    # Load data
    if len(pickle.loads(binary_data[0]["design_params"])) != 0:
        WITH_DESIGN_PARAMS = True
        data = [[pickle.loads(binary_data[i]["W_0"]), pickle.loads(binary_data[i]["dfree_0"]),  
                        pickle.loads(binary_data[i]["s_a"]), pickle.loads(binary_data[i]["s_e"]),
                        pickle.loads(binary_data[i]["W"]), pickle.loads(binary_data[i]["dfree"]),
                        pickle.loads(binary_data[i]["design_params"])] for i in range(len(binary_data))]

        columns_labels = ["W_0", "dfree_0", "s_a", "s_e", "W_t", "dfree_t", "design_params"]

    else:
        data = [[pickle.loads(binary_data[i]["W_0"]), pickle.loads(binary_data[i]["dfree_0"]),  
                        pickle.loads(binary_data[i]["s_a"]), pickle.loads(binary_data[i]["s_e"]),
                        pickle.loads(binary_data[i]["W"]), pickle.loads(binary_data[i]["dfree"])] 
                        for i in range(len(binary_data))]
        columns_labels = ["W_0", "dfree_0", "s_a", "s_e", "W_t", "dfree_t"]
        

    # Turn dataset into a panda dataframe
    final_labels = []
    for c in range(len(columns_labels)):       
        # Case vector
        if np.array(data[0][c]).ndim == 1:
            for i in range(np.array(data[0][c]).shape[0]):
                final_labels.append(columns_labels[c] + "_" + str(i))
        # Case matrice
        elif data[0][c].ndim == 2:
            for i in range(data[0][c].shape[0]):
                for j in range(data[0][c].shape[1]):
                    final_labels.append(columns_labels[c] + "_" + str(i) + "_" + str(j))
        
    data_to_df = []
    for l in range(len(data)):
        data_line = []
        for c in range(len(data[l])):
            # Case vector
            if np.array(data[l][c]).ndim == 1:
                for i in range(np.array(data[l][c]).shape[0]):
                    data_line.append(data[l][c][i])           
            # Case matrice
            elif data[l][c].ndim == 2:
                for i in range(data[l][c].shape[0]):
                    for j in range(data[l][c].shape[1]):
                        data_line.append(data[l][c][i][j])      
        data_to_df.append(data_line)

    df = pd.DataFrame(data_to_df, columns = final_labels)
    df = df.astype(np.float16)

    ### Searching for correlation between all data
    if analysis_mode == "EXPLORE_DATA":
        # Show dataframe in a tabloo window
        #tabloo.show(df)

        ### Searching for correlations between variables
        corr_df = df.corr()
        tabloo.show(corr_df)

        # Test specific to Finger contact case
        # physics_informed_loss = compute_true_physical_informed_value_loss(data, config)

        # Plot all correlations
        plt.figure(figsize=(30,30))
        sns.heatmap(corr_df,
                xticklabels=corr_df.columns.values,
                yticklabels=corr_df.columns.values,
                vmin=-1.0, vmax=1.0,
                cmap='RdYlGn_r')
        plt.show()


    ### Showing basic stats on data
    if analysis_mode == "EXPLORE_DATA_RANGE":
        # Compute and show basic statistics on acquired data in a table
        df_stats = df.describe().transpose()
        df_stats.insert(loc=0, column='attribute', value=final_labels)
        # tabloo.show(df_stats)

        # Compute boxplot by type of data
        for i in range(len(columns_labels)):
            corr_final_labels = [final_labels[k] for k in range(len(final_labels)) if columns_labels[i] in final_labels[k]]
            if not len(corr_final_labels) == 0:
                ax = df[corr_final_labels].plot(kind='box', title='boxplot', showmeans=True, sharey=False)
        plt.show()

    ### Correlation plot betwene two specific data
    if analysis_mode == "EXPLORE_DATA_CORRELATION":
        LABEL_X = "design_params_0"
        LABEL_Y = "design_params_1"
        LABEL_Z = "W_0_1_1"

        # LABEL_X = "design_params_0"
        # LABEL_Y = "W_0_1_1"
        # LABEL_Z = None

        x = df[LABEL_X].values.tolist()
        y = df[LABEL_Y].values.tolist()
        if LABEL_Z is not None:
            z = df[LABEL_Z].values.tolist()

        fig = plt.figure()
        if LABEL_Z is not None:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
        if LABEL_Z is not None:
            ax.scatter(x, y, z)
        else:
            ax.scatter(x, y)
        ax.set_xlabel(LABEL_X)
        ax.set_ylabel(LABEL_Y)
        if LABEL_Z is not None:
            ax.set_zlabel(LABEL_Z)
        plt.show()

        
    ### Display the sampling workspace
    if analysis_mode in ["EXPLORE_DATA_WORKSPACE_TRAINING", "EXPLORE_DATA_WORKSPACE_TEST"]:
        MODE_QUIVER = False

        n_act = len(config.get_actuators_variables())
        n_contact = len(config.get_contacts_variables())
        n_design = len(config.get_design_variables())

        points_3D = []
        orientations_3D = []
        for i in range(len(data)):
            
            if n_contact == 0:
                points = data[i][5][n_act:]
            else:
                points = data[i][2][n_act:]
    
            # For now we take the average of all constraints points as the 3D pos
            nb_points = int(len(points) / 3)
            res = np.zeros(3)
            for i in range(nb_points):
                res += np.array(points[3*i : 3*(i+1)]) 
            res /= nb_points
            points_3D.append(res.tolist())


            # Compute orientation of the plane if relevant
            if nb_points == 3:
                MODE_QUIVER = True
                A = np.array(points[0 : 3])
                B = np.array(points[3 : 6])
                C = np.array(points[6 : 9])
                direction = np.cross(B - A, C - A)
                norm = direction / len(direction)
                orientations_3D.append(norm.tolist())

        points_3D = np.array(points_3D)
        orientations_3D = np.array(orientations_3D)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = points_3D[:,0]
        Y = points_3D[:,1]
        Z = points_3D[:,2]
        if not MODE_QUIVER:
            ax.scatter3D(X, Y, Z, color = "green")
        else:
            U = orientations_3D[:,0]
            V = orientations_3D[:,1]
            W = orientations_3D[:,2]
            ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
        plt.show()


    ### Exploring our ability to learn mechanical matrices without application
    if analysis_mode in ["EXPLORE_LEARNING_OVERFITTING", "EXPLORE_LEARNING_VALIDATION", 
                         "EXPLORE_LEARNING_OVERFITTING_DESIGN_PIPELINE", "EXPLORE_LEARNING_VALIDATION_DESIGN_PIPELINE"]:
        import torch
        utils_lib = importlib.import_module("Libraries.utils")
        MLP_lib = importlib.import_module("Libraries.Learning.MLP.learning_tools")

        # Choose network - TODO: add network laoder based on chosen sampling stats
        config.config_network = ask_user_NN_properties(config.model_name, config.config_network["network_name"])        
        config.config_network.update({"learning_rate":0.00001, "dropout_probability": 0.0})

        ### Load network(s)
        model, _, _, _, _, n_constraint, scaling, _, _, last_model_link, best_model_link, _ = utils_lib.init_network(config.config_network["network_name"], config)
        
        if pathlib.Path.exists(best_model_link):
            checkpoint = torch.load(best_model_link)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['loss']
            print(">>   Reload the best model from epoch {} with test loss {}".format(best_epoch, best_test_loss))
        else:
            print("[ERROR]  >>  No file to load the best model.")
            exit(1)

        if WITH_DESIGN_PARAMS:
            config.config_network = ask_user_NN_properties(config.model_name, config.config_network["network_name"])        
            config.config_network.update({"learning_rate":0.00001, "dropout_probability": 0.0})
            model_design, _, _, _, _, n_constraint_design, scaling_design, _, _, last_model_link_design, best_model_link_design, _ = utils_lib.init_network(config.config_network["network_name"], config, design_to_MM = True)
            if pathlib.Path.exists(best_model_link_design):
                checkpoint = torch.load(best_model_link_design)
                #checkpoint = torch.load(last_model_link_design)
                model_design.load_state_dict(checkpoint['model_state_dict'])
                best_epoch = checkpoint['epoch']
                best_test_loss = checkpoint['loss']
                print(">>   Reload the best model for design from epoch {} with test loss {}".format(best_epoch, best_test_loss))
            else:
                print("[ERROR]  >>  No file to load the best design model.")
                exit(1)


        ### Let's test if we can Overfit training data
        # Init sum of errors
        W_0 = data[i][0]
        dfree_0 = data[i][1]
        sum_W = np.subtract(W_0, W_0)
        sum_dfree = np.subtract(dfree_0, dfree_0)
        list_error_W = []
        list_error_dfree = []

        if WITH_DESIGN_PARAMS:
            sum_W_0 = np.subtract(W_0, W_0)
            sum_dfree_0 = np.subtract(dfree_0, dfree_0)
            list_error_W_0 = []
            list_error_dfree_0 = []

        for i in range(len(data)):
            ### Get data from dataset
            W_0 = data[i][0]
            dfree_0 = data[i][1]
            s_a = data[i][2]
            s_e = data[i][3]

            W_t = data[i][4]
            dfree_t = data[i][5]

            if WITH_DESIGN_PARAMS:
                design_params = data[i][6]

            normalization_method = config.config_network["data_normalization"]
            
            ### Prediction of initial mechanical matrices from design parameters
            if WITH_DESIGN_PARAMS:
                # Rescale for design params
                X_design = [torch.tensor(design_params)]
                Y_design = [torch.tensor(W_0), torch.tensor(dfree_0)]
                
                if normalization_method == "Std":
                    X_design, _ = MLP_lib.create_data_std(X_design, Y_design, scaling_design[0], scaling_design[1], scaling_design[2], scaling_design[3], design_to_MM = True)
                elif normalization_method == "MinMax":
                    X_design, _ = MLP_lib.create_data_minmax(X_design, Y_design, scaling_design[0], scaling_design[1], scaling_design[2], scaling_design[3], design_to_MM = True)
                else:
                    X_design, _ = MLP_lib.create_data(X_design, Y_design, design_to_MM = True)
            
                # Prediction
                Y_design = model_design(X_design).detach().numpy()[0]
            
            ### Prediction of mechanical matrices from initial mechanical matrices and actuation displacement
            # Rescale for mechanical matrices
            if analysis_mode in ["EXPLORE_LEARNING_OVERFITTING_DESIGN_PIPELINE", "EXPLORE_LEARNING_VALIDATION_DESIGN_PIPELINE"]:
                
                # Rebuild dfree_0 and W_0
                dfree_0_pred = Y_design[-n_constraint:]
                W_pred = Y_design[:-n_constraint]
                W_0_pred = np.zeros((n_constraint, n_constraint))
                W_0_pred[np.triu_indices(n=n_constraint)] = W_pred
                W_0_pred[np.tril_indices(n=n_constraint, k=-1)] = W_0_pred.T[np.tril_indices(n=n_constraint, k=-1)]
                W_0_pred = W_0_pred.reshape(-1)
                
                # Rescale to come back to real value
                if normalization_method == "Std":
                    dfree_0_pred = dfree_0_pred * (scaling_design[3][1]) + scaling_design[2][1] 
                elif normalization_method == "MinMax":  
                    dfree_0_pred = dfree_0_pred * (scaling_design[3][1]  - scaling_design[2][1]) + scaling_design[2][1]
                
                if normalization_method == "Std":
                    W_0_pred = W_0_pred * (scaling_design[3][0].reshape(-1)) + scaling_design[2][0].reshape(-1) 
                elif normalization_method == "MinMax":
                    W_0_pred = W_0_pred * (scaling_design[3][0].reshape(-1) - scaling_design[2][0].reshape(-1)) + scaling_design[2][0].reshape(-1) 
                W_0_pred = W_0_pred.reshape(n_constraint,n_constraint)
                
                # Build X vector
                X = [torch.tensor(W_0_pred), torch.tensor(dfree_0_pred), torch.tensor(s_a + [0]*len(s_e))]
            else:
                X = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a + [0]*len(s_e))]
            Y = [torch.tensor(W_0), torch.tensor(dfree_0)]
            
            if normalization_method == "Std":
                X, _ = MLP_lib.create_data_std(X, Y, scaling[0], scaling[1], scaling[2], scaling[3])
            elif normalization_method == "MinMax":
                X, _ = MLP_lib.create_data_minmax(X, Y, scaling[0], scaling[1], scaling[2], scaling[3])
            else:
                X, _ = MLP_lib.create_data(X, Y)
                
            # Prediction
            Y = model(X)
                
            ### Rescale outputs
            # Prediction robot state from initial state
            dfree = Y[-n_constraint:].detach().numpy()
            if normalization_method == "Std":
                dfree = dfree * (scaling[3][1]) + scaling[2][1] # Rescale dfree
            elif normalization_method == "MinMax":
                dfree = dfree * (scaling[3][1]  - scaling[2][1]) + scaling[2][1] # Rescale dfree

            W_pred = Y[:-n_constraint].detach().numpy()
            W = np.zeros((n_constraint, n_constraint))
            W[np.triu_indices(n=n_constraint)] = W_pred
            W[np.tril_indices(n=n_constraint, k=-1)] = W.T[np.tril_indices(n=n_constraint, k=-1)]
            W = W.reshape(-1)
            if normalization_method == "Std":
                W = W * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
            elif normalization_method == "MinMax":
                W = W * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
            W = W.reshape(n_constraint,n_constraint)

            # Prediction initial state from design parameters
            if WITH_DESIGN_PARAMS:
                dfree_0_pred = Y_design[-n_constraint:]
                W_pred = Y_design[:-n_constraint]

                if normalization_method == "Std":
                    dfree_0_pred = dfree_0_pred * (scaling_design[3][1]) + scaling_design[2][1] 
                elif normalization_method == "MinMax":  
                    dfree_0_pred = dfree_0_pred * (scaling_design[3][1]  - scaling_design[2][1]) + scaling_design[2][1]

                W_0_pred = np.zeros((n_constraint, n_constraint))
                W_0_pred[np.triu_indices(n=n_constraint)] = W_pred
                W_0_pred[np.tril_indices(n=n_constraint, k=-1)] = W_0_pred.T[np.tril_indices(n=n_constraint, k=-1)]
                W_0_pred = W_0_pred.reshape(-1)

                if normalization_method == "Std":
                    W_0_pred = W_0_pred * (scaling_design[3][0].reshape(-1)) + scaling_design[2][0].reshape(-1) 
                elif normalization_method == "MinMax":
                    W_0_pred = W_0_pred * (scaling_design[3][0].reshape(-1) - scaling_design[2][0].reshape(-1)) + scaling_design[2][0].reshape(-1) 
                W_0_pred = W_0_pred.reshape(n_constraint,n_constraint)

            ### Display element-wise results
            if WITH_DESIGN_PARAMS:
                print("Ground Truth dfree_0:", dfree_0)
                print("Prediction dfree_0:", dfree_0_pred)

                print("Ground Truth W_0:", W_0)
                print("Prediction W_0:", W_0_pred)

            print("Ground Truth dfree_t:", dfree_t)
            print("Prediction dfree_t:", dfree)

            print("Ground Truth W_t:", W_t)
            print("Prediction W_t:", W)

            ### Compute error metrics
            # Compute element-wise prediction error metric
            error_W = np.abs(np.subtract(W, W_t))
            error_dfree = np.abs(np.subtract(dfree, dfree_t))
            if WITH_DESIGN_PARAMS:
                error_W_0 = np.abs(np.subtract(W_0_pred, W_0))
                error_dfree_0 = np.abs(np.subtract(dfree_0_pred, dfree_0))

            # Update sums of errors for computing the mean
            sum_W += error_W
            sum_dfree = error_dfree
            if WITH_DESIGN_PARAMS:
                sum_W_0 += error_W_0
                sum_dfree_0 = error_dfree_0

            # Update list of numpy error arrays for computing standard deviation
            list_error_W.append(error_W)
            list_error_dfree.append(error_dfree)
            if WITH_DESIGN_PARAMS:
                list_error_W_0.append(error_W_0)
                list_error_dfree_0.append(error_dfree_0)

        ### Print indicators
        if WITH_DESIGN_PARAMS:
            print("Averaged error on W_0 prediction:", sum_W_0 / len(data))
            print("Averaged error on dfree_0 prediction:", sum_dfree_0 / len(data))
            
            print("Standard deviation error on W_0 prediction:", np.std(np.dstack(list_error_W_0), 2, ddof=1))
            print("Standard deviation error on dfree_0 prediction:", np.std(np.dstack(list_error_dfree_0), 2, ddof=1))

            print("Worst error on W_0 prediction:", np.array(list_error_W_0).max(axis=0))
            print("Worst error on dfree_0 prediction:", np.array(list_error_dfree_0).max(axis=0))

        print("Averaged error on W prediction:", sum_W / len(data))
        print("Averaged error on dfree prediction:", sum_dfree / len(data))
        
        print("Standard deviation error on W prediction:", np.std(np.dstack(list_error_W), 2, ddof=1))
        print("Standard deviation error on dfree prediction:", np.std(np.dstack(list_error_dfree), 2, ddof=1))

        print("Worst error on W prediction:", np.array(list_error_W).max(axis=0))
        print("Worst error on dfree prediction:", np.array(list_error_dfree).max(axis=0))



def compute_true_physical_informed_value_loss(data, config):
    loss = []
    n_act_constraint = len(config.get_actuators_variables())
    is_contact = (len(config.get_contacts_variables()) != 0)
    n_constraint = len(data[0][2])

    for i in range(len(data)):
        if len(data[i]) == 6:
            [_, _, s, s_e, W_t, dfree_t] = data[i]
        elif len(data[i]) == 7:
            [_, _, s, s_e, W_t, dfree_t, _] =  data[i]

        if is_contact:
            # Compute the state of the constraint (delta)
            s_a = np.array(s[:n_act_constraint])
            s_c = np.array(s[n_act_constraint:])

            # Compute the dfree
            dfree_a = dfree_t[:n_act_constraint]
            dfree_c = dfree_t[n_act_constraint:]

            Waa = W_t[:n_act_constraint, :n_act_constraint]
            Wac = W_t[:n_act_constraint, n_act_constraint:]
            Wca = W_t[n_act_constraint:, :n_act_constraint]
            Wcc = W_t[n_act_constraint:, n_act_constraint:]

            # Compute the physics inform term of the loss
            Ds_a = np.expand_dims((s_a - dfree_a), axis=1)
            Ds_c = np.expand_dims((s_c - dfree_c), axis=1)

            invWaa = np.linalg.inv(Waa)
            M = Wcc - np.matmul(Wca, np.matmul(invWaa, Wac))
            invM = np.linalg.inv(M)
            lambda_c = np.matmul(invM, Ds_c) - np.matmul(invM, np.matmul(Wca, np.matmul(invWaa, Ds_a)))
            lambda_a = np.matmul(invWaa, Ds_a) - np.matmul(invWaa, np.matmul(Wac, lambda_c))

            # invWcc = np.linalg.inv(Wcc)
            # M = Waa - np.matmul(Wac, np.matmul(invWcc, Wca))
            # invM = np.linalg.inv(M)
            # lambda_a = np.matmul(invM , Ds_a) - np.matmul(invM, np.matmul(Wac, np.matmul(invWcc, Ds_c)))
            # lambda_c = np.matmul(invWcc, Ds_c) - np.matmul(invWcc, np.matmul(Wca, lambda_a))

            physics_informed_term_c = np.matmul(Wcc, lambda_c) + np.matmul(Wca, lambda_a) - Ds_c
            physics_informed_term_a = np.matmul(Wac, lambda_c) + np.matmul(Waa, lambda_a) - Ds_a

            print("\n")
            print("Avec c:", physics_informed_term_c)
            print("Avec a:", physics_informed_term_a)
            print("\n")

        else:
            # Compute the state of the constraint (delta)
            s_a = np.array(s)
            s_e = np.array(s_e)
            print("s_e:", s_e)

            # Compute the dfree
            dfree_a = dfree_t[:n_act_constraint]
            dfree_e = dfree_t[n_act_constraint:]

            Waa = W_t[:n_act_constraint, :n_act_constraint]
            Wae = W_t[:n_act_constraint, n_act_constraint:]
            Wea = W_t[n_act_constraint:, :n_act_constraint]
            Wee = W_t[n_act_constraint:, n_act_constraint:]

            # Compute the physics inform term of the loss
            Ds_a = np.expand_dims((s_a - dfree_a), axis=1)
            Ds_e = np.expand_dims((s_e - dfree_e), axis=1)

            invWaa = np.linalg.inv(Waa)
            physics_informed_term = np.matmul(Wea, np.matmul(invWaa, Ds_a)) - Ds_e

            print("physics_informed_term:", physics_informed_term)


def ask_user_analysis_mode(list_modes):
        
    print("Available data analysis modes are:")
    for i, mode in enumerate(list_modes):
        print(">> Analysis mode n° ", i, ":", mode)

    user_input_model = int(input("What is the number of the data analysis mode you want to use ?"))
   
    while user_input_model<0 or user_input_model>i:
        print("Please answer a number in:", [0, i])
        user_input_model = int(input("What is the number of the data analysis mode you want to use ?"))

    return list_modes[user_input_model]
    