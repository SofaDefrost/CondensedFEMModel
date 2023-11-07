# -*- coding: utf-8 -*-
"""Main file to launch script (learning, data acquisition, applications).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

# System libs
import argparse
import pathlib
import importlib
from Libraries.database import query_learning_stats, query_sampling_stats_for_a_model, add_learning_stats


def main(args=None):
    """ Main entry point for the project
    Parameters
    ----------
    args : list
        A list of arguments as if they were input in the command line. Leave it to None to use sys.argv.
    """

    # Parse arguments
    parser = argparse.ArgumentParser('Process args')
    parser.add_argument('--name', '-n', help='Load a model: -n model_name')
    parser.add_argument('--app', '-app', help='Application from learned model: -app application_name')

    # Hyperparameters for data acquisition   
    """ 
    sampling_strategy: str in {Grid, }
        Name of used samplign strategy. Available samplign strategy are:
            - Random: sample points at random.
            - Grid: exhaustively sample actuation space.
            - SH: Scrambled Halton sampling i.e. low discrepancy sequence with scrambling to ameliorate Halton sequence by limitating severe striping artifacts
            - LHS: Latin Hypercube Sampling generates a set of samples such that each sample point is formed by randomly selecting one coordinate value from a grid from each dimension
            - LCVT: Latinized Centroidal Voronoi Tessellation as described in "Smart sampling and incremental function learning for very large high dimensional data"
    n_samples: int
        Number of considered samples
    """
    parser.add_argument("--sampling_strategy", "-ss", help="The sampling strategy for data acquisition.", type=str, default= "SH")
    parser.add_argument("--n_samples", "-ns", help="The maximum number of samples considered during data acquisition.", type=int, default=100)

    # Hyperparameters tuning for training a model
    """
    network_name: str in {MLP, doubleMLP}
        Name of used neural network. Available networks are:
            - MLP: low memory NN, dependant to input and output mechanical matrices size 
            - doubleMLP: two parallele MLP: one for W and one for dfree
    mode_loss: str
        The loss used for training the network. Available loss are:
            - Euclidean: Sum of element-wise eucliden distance between predicted and data acquired matrices.
            - MSE
            - L1
            - RSE 
            - physics_informed: MSE + physics informed loss
            - MSEregL1: MSE + regularization L1 
            - MSEregL2: MSE + regularization L2
    n_hidden_layers: int
        Number of hidden layers in the network. TODO: add advised choices for MLP.
    latent_size: int
        Size of hidden layers. TODO: add advised choices for MLP.
    batch_size: int
        Size of a btches used for training NN.
    data_normalization: str
        Strategy used for normalizing data. It is necessary as projected compliance matrice elements can have very small values.
        The avaialable strategies are:
            - None: No strategy is used
            - MinMax: Use minimum and maximum value for normalizing each feature
            - Std: Use mean and standard deviation for normalizing each feature
    """
    parser.add_argument("--network_name", "-nn", help="The name of the neural network we want to use.", type=str, default= "MLP")
    parser.add_argument( "--mode_loss", "-ml", help="The name of the loss we want to use.", type=str, default="MSE") 
    parser.add_argument("--n_hidden_layers", "-nh", help="The number of hidden layers.", type=int, default=3)
    parser.add_argument("--latent_size", "-hs", help="The size of hidden layers.", type=int, default = 450) #400 500 900
    parser.add_argument("--batch_size", "-bs", help="The batch size.", type=int, default = 2048) 
    parser.add_argument("--learning_rate", "-lr", help="The learning rate.", type=float, default = 1e-3)
    parser.add_argument("--data_normalization", "-dn", help="The data normalization strategy used", type=str, default = "Std")
    parser.add_argument("--dropout_probability", "-dp", help="The dropout probability.", type=float, default = 0.0)
    parser.add_argument("--design_to_MM", "-dtMM", help="Use a MLP to learn (W_0, dfree_0) from design parameters.", action='store_true')


    parser.add_argument("--epochs", "-ne", help="The number of epochs for the training.", type=int, default = 50000)
    
    parser.add_argument("--n_cores", "-nc", help="The number of cores in the machine.", type=int, default = 1)
    parser.add_argument('--load_optimisation', '-lo', help='Load the best params of the optimisation', action='store_true')


    # How to test the scenes (in app == TestQP)
    """
    type_use: str in [learned, simulated, interpolated]
        To use learned matrices, compute them or interpolate them
    use_trajectory: boolean
        To use a provided trajectory for evaluating the design.
    """
    parser.add_argument("--type_use", "-tu", help="Use learned, computed or interpolated matrices.", type=str, default= "simulated")
    parser.add_argument('--use_trajectory', '-ut', help='Use trajectory for evaluating the design', action='store_false')
    

    args = parser.parse_args(args)
    assert args.name != None, "Please enter a model name"
    config_link = pathlib.Path(str(pathlib.Path(__file__).parent.absolute())+"/Models/"+ args.name+"/Config.py")
    print("Config link", config_link)
    assert pathlib.Path.exists(config_link), "Please enter a valid model name"

    # Main operand
    config_lib = importlib.import_module("Models."+ args.name+".Config")
    Config = config_lib.Config()
    ratio_test_train = 0.25

    if args.app in ["Training" , "TestQP", "Lplot",  "TestWFromD", "DataAn",  "NHO", "pS2R"]:
        if args.name == "2Finger":
            model_name = "Finger" 
        else:
            model_name = args.name

        config_network = {"model_name": model_name,
                           "sampling_strategy": args.sampling_strategy,
                           "n_samples": args.n_samples,
                            "network_name": args.network_name, 
                            "mode_loss": args.mode_loss, 
                            "n_hidden_layers": args.n_hidden_layers,
                            "latent_size": args.latent_size,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "data_normalization": args.data_normalization, 
                            "dropout_probability": args.dropout_probability}

        if args.app in ["Training", "Lplot", "pS2R"]:
            try:
                registered_learning_stat = query_learning_stats(config_network["model_name"], 
                                                                config_network["sampling_strategy"],
                                                                config_network["n_samples"],
                                                                config_network["network_name"], 
                                                                config_network["mode_loss"], 
                                                                config_network["n_hidden_layers"], 
                                                                config_network["latent_size"], 
                                                                config_network["batch_size"],
                                                                config_network["data_normalization"])
                config_network["sampling_strategy"] = params["sampling_strategy"]
                config_network["n_samples"] = params["n_samples"]
                
            except:
                query_SS_properties = list(query_sampling_stats_for_a_model(config_network["model_name"]))
                
                print("Sampling strategies parameters for the model ", config_network["model_name"], ":")
                for i, params in enumerate(query_SS_properties):
                    print(">> Sampling strategy n° ", i, ":", query_SS_properties[i])

                user_input_model = int(input("Which samplign strategy do you want to use ? (give the number of the SS in the previous list) "))
                while user_input_model<0 or user_input_model>i:
                    print("Please answer a number in:", [0, i])
                    user_input_model = int(input("Which samplign strategy do you want to use ? (give the number of the SS in the previous list) "))

                params = query_SS_properties[user_input_model]
                config_network["sampling_strategy"] = params["sampling_strategy"]
                config_network["n_samples"] = params["n_samples"]

                registered_learning_stat = query_learning_stats(config_network["model_name"], 
                                                                config_network["sampling_strategy"],
                                                                config_network["n_samples"],
                                                                config_network["network_name"], 
                                                                config_network["mode_loss"], 
                                                                config_network["n_hidden_layers"], 
                                                                config_network["latent_size"], 
                                                                config_network["batch_size"],
                                                                config_network["data_normalization"])

            if len(registered_learning_stat) == 0:  # Create a new learnign strategy
                id_learning_stat = add_learning_stats(config_network["model_name"], 
                                                    config_network["sampling_strategy"],
                                                    config_network["n_samples"],
                                                    config_network["network_name"], 
                                                    config_network["mode_loss"], 
                                                    config_network["n_hidden_layers"], 
                                                    config_network["latent_size"], 
                                                    config_network["batch_size"],
                                                    config_network["data_normalization"])
            else: # Else load an existing one
                id_learning_stat = registered_learning_stat[0]["id"]
            config_network.update({"id": id_learning_stat})
        Config.config_network = config_network

    if args.app:
        #assert args.app == "Test" or args.app == "Lplot" or args.app == "TestQP", "Please enter a valid application name"
        if args.app ==  "AcquireData":
            print("Starting data acquisition")
            data_lib = importlib.import_module("Applications.AcquireData")
            data_lib.acquire_data(Config, n_train_samples=args.n_samples, ratio_test_train=ratio_test_train,
                                  n_process=20, train_sampling_strategy=args.sampling_strategy)
        if args.app == "Training": # A simple function to train the networks
            learning_lib = importlib.import_module("Applications.Training")
            assert config_network["network_name"] == "MLP" or config_network["network_name"] == "doubleMLP"
            print("Starting learning from data (", config_network["network_name"], " model)")
            learning_lib.train_network(config_network["network_name"], Config, n_train_samples = args.n_samples, ratio_test_train = ratio_test_train, epochs = args.epochs, use_GPU = False, design_to_MM = args.design_to_MM)
        if args.app == "Test": # A simple function to test the baseline inverse scene
            test_simu_lib = importlib.import_module("Applications.TestSimu")
            test_simu_lib.test_simu(Config, is_inverse = False, is_force = False)
        if args.app == "TestQP": # A simple function to test the baseline inverse scene
            test_simu_lib = importlib.import_module("Applications.TestQPGUI")
            test_simu_lib.test_simu_qp(Config, network_name = config_network["network_name"], type_use = args.type_use, use_trajectory = args.use_trajectory)
        if args.app == "Lplot": # Plot the loss curve evolution for the test set during learning
            learning_lib = importlib.import_module("Learning.Training")
            learning_lib.plot_test_loss(Config)
        if args.app == "pS2R": # Prepare the files for sim2Real. Note: only for MLP and pneumaticTrunk
            prepare_sim2Real_lib = importlib.import_module("Applications.PrepareForReality")
            prepare_sim2Real_lib.prepareForReality(config_network["network_name"], Config, n_train_samples = args.n_samples, ratio_test_train = ratio_test_train, epochs = args.epochs, use_GPU = False, design_to_MM = args.design_to_MM)

        if args.app == "NHO": # Optimize hyperparameters of a NN
            print("Starting optimisation of the neural network hyperparameters.")
            raise Exception("Sorry, this component is not available right now.") 
            optimisation_lib = importlib.import_module("Applications.NetworkHyperparametersOptimisation")
            optimisation_lib.hyperparameters_optimisation(Config, n_cores = args.n_cores, load = args.load_optimisation, n_optimisation = 1000)

        if args.app == "DO": # Design optimization assisted by surrogate
            design_optimization_lib = importlib.import_module("Applications.DesignOptimization")
            design_optimization_lib.design_optimization(Config)

        if args.app == "DataAn": # Data analysis of acquired data
            data_analysis_lib = importlib.import_module("Applications.DataAnalysis")
            data_analysis_lib.data_analysis(Config, design_to_MM = args.design_to_MM)
            
        if args.app == "TestWFromD": # Test W prediction from NN trained on design parameters
            print("Although this part of the code is working, it will be soon moved in the Data Analysis script.")
            test_W_from_design_lib = importlib.import_module("Applications.GetWFromDesignParams")
            test_W_from_design_lib.W_from_design_params(Config, network_name = config_network["network_name"], type_use = "comparison")

        
        
        
if __name__ == "__main__":
    main()
