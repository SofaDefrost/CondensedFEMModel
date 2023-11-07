import json
import numpy as np

#For the elasticity parameters
name_ground_truth = "./Data/trajectory_FingerFineFine_computed.txt"
name_finger = "./Data/trajectory_Finger_learned.txt"
name_fingerfine = "./Data/trajectory_FingerFine_learned.txt"
name_fingerfinefine = "./Data/trajectory_FingerFineFine_learned.txt"

def manage_data(name, type_data = "pos_effector"):
    with open(name, 'r') as fp:
        load_data = json.load(fp)
        data = load_data[type_data]

    output_data = []
    for i in range(len(data["x"])):
        output_data.append([data["x"][i], data["y"][i], data["z"][i]])

    return output_data


#Mechanical Parameters
idx = [0, 1, 2, 3]
YoungsModulus = [5000, 1000, 7000, 3000]
PoissonRatio = [0.47, 0.45 ,0.4, 0.47]
Length = [40]*4
Height = [20]*4
JointHeight = [6]*4


for i in idx:
    name_learned = "./Data/trajectory_FingerElasticityParams_learned_" + str(i)+".txt"
    name_computed = "./Data/trajectory_FingerElasticityParams_computed_" + str(i)+".txt"

    data_goal = manage_data(name_computed, type_data = "pos_goal")
    data_computed = manage_data(name_computed, type_data = "pos_effector")
    data_learned = manage_data(name_learned, type_data = "pos_effector")

    nb_data_goal = len(data_goal)
    nb_data_computed = len(data_computed)
    nb_data_learned = len(data_learned)

    error_computed, error_learned = 0, 0
    for j in range(nb_data_goal):
        pos_goal = np.array(data_goal[j])
        pos_computed = np.array(data_computed[j])
        pos_learned = np.array(data_learned[j])

        error_computed += np.linalg.norm(pos_goal- pos_computed)/nb_data_goal
        error_learned += np.linalg.norm(pos_goal- pos_learned)/nb_data_goal

    print("[INFO, Mechanical] Finger parameters:")
    print(">> Young Modulus:", YoungsModulus[i])
    print(">> Poisson Ratio:", PoissonRatio[i])
    print(">> Length:", Length[i])
    print(">> Height:", Height[i])
    print(">> JointHeight:", JointHeight[i])
    print("[INFO] Results:")
    print(">> Mean error for the computed: ", error_computed)
    print(">> Mean error for the learned: ", error_learned)
    print(">> Percent: ", abs(error_computed - error_learned)/error_computed)
    print("\n")



#Geometrical Parameters
idx = [0, 1, 2, 3]
YoungsModulus = [3000]*4
PoissonRatio = [0.47]*4
Length = [38.5, 40, 41, 39.5]
Height = [21.5, 20.5, 21, 20]
JointHeight = [6, 7.5, 5.5, 7]

for i in idx:
    name_learned = "./Data/trajectory_FingerDesign_learned_" + str(i)+".txt"
    name_computed = "./Data/trajectory_FingerDesign_computed_" + str(i)+".txt"

    data_goal = manage_data(name_computed, type_data = "pos_goal")
    data_computed = manage_data(name_computed, type_data = "pos_effector")
    data_learned = manage_data(name_learned, type_data = "pos_effector")

    nb_data_goal = len(data_goal)
    nb_data_computed = len(data_computed)
    nb_data_learned = len(data_learned)

    error_computed, error_learned = 0, 0
    for j in range(nb_data_goal):
        pos_goal = np.array(data_goal[j])
        pos_computed = np.array(data_computed[j])
        pos_learned = np.array(data_learned[j])

        error_computed += np.linalg.norm(pos_goal- pos_computed)/nb_data_goal
        error_learned += np.linalg.norm(pos_goal- pos_learned)/nb_data_goal

    print("[INFO, Geometrical] Finger parameters:")
    print(">> Young Modulus:", YoungsModulus[i])
    print(">> Poisson Ratio:", PoissonRatio[i])
    print(">> Length:", Length[i])
    print(">> Height:", Height[i])
    print(">> JointHeight:", JointHeight[i])
    print("[INFO] Results:")
    print(">> Mean error for the computed: ", error_computed)
    print(">> Mean error for the learned: ", error_learned)
    print(">> Percent: ", abs(error_computed - error_learned)/error_computed)
    print("\n")
