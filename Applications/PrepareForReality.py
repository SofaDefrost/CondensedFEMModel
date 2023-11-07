import shutil
import json
import sys
import pathlib
import importlib
import numpy as np
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
learning_lib = importlib.import_module("Applications.Training")

def prepareForReality(network_name, config, n_train_samples = 10000, ratio_test_train = 0.25, epochs = 3000, use_GPU = False, design_to_MM = False):
    path = str(pathlib.Path(__file__).parent.absolute())+"/../EmbeddedControl"

    models_name = ["CoarsePneumaticTrunk1Effector", "CoarsePneumaticTrunk", "MediumPneumaticTrunk", "FinePneumaticTrunk"]
    if not(config.model_name in models_name):
        print("[ERROR] >> The sim2Real files are generated for a robot in ", models_name, " not for ", config.model_name, ".")
        exit(1)

    if network_name != "MLP":
        print("[ERROR] >> The sim2Real files are generated for  MLP not for ", network_name, ".")
        exit(2)

    batch_size = config.config_network["batch_size"]
    _, _, _, dataloader, _, n_constraint, scaling, _, _, _, best_model_link, _ = learning_lib.init_network(
        network_name, config, batch_size, ratio_test_train, use_GPU=use_GPU, design_to_MM=design_to_MM)

    print(best_model_link)
    os.makedirs(path + "/Data", exist_ok=True)
    if pathlib.Path.exists(best_model_link):
        shutil.copy(best_model_link, path + "/Data/model.pth")

        LATENT_SIZE = config.config_network["latent_size"]
        N_HIDDEN_LAYER = config.config_network["n_hidden_layers"]
        OUTPUT_SIZE = len(np.triu_indices(n=n_constraint)[0]) + n_constraint
        INPUT_SIZE = OUTPUT_SIZE + n_constraint

        network_param = [LATENT_SIZE, N_HIDDEN_LAYER, OUTPUT_SIZE, INPUT_SIZE]
        for data in dataloader:
            X_batch, _ = data
            W0, dfree0 = X_batch[0][0].tolist(), X_batch[1][0].tolist()
            break

        with open( path + "/Data/network_parameters.txt", 'w') as fp:
            json.dump(network_param, fp)

        with open( path + "/Data/init_data.txt", 'w') as fp:
            json.dump([W0, dfree0], fp)

        scaling[0] = [s.tolist() for s in scaling[0]]
        scaling[1] = [s.tolist() for s in scaling[1]]
        scaling[2] = [s.tolist() for s in scaling[2]]
        scaling[3] = [s.tolist() for s in scaling[3]]
        with open( path + "/Data/scaling.txt", 'w') as fp:
            json.dump(scaling, fp)

    else:
        print("[ERROR] >> Best model link doesn't exist. Please load an existing model.")