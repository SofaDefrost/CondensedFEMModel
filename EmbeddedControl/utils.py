import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import numpy as np
import proxsuite
import sys
import pathlib
class MLP(nn.Module):
    """Classical MLP.
    """
    def __init__(self, input_size, output_size, latent_size, n_hidden_layers = 1, dropout_probability = 0):
        """Initialization.

        Parameters:
        -----------
            input_size: int
                The size of the input.
            output_size: int
                The size of the output.
            latent_size: int
                The size of all the hidden layers.
            n_hidden_layers: int, default = 1
                The number of hidden layers in the MLP.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.n_hidden_layers = n_hidden_layers

        self.dropout_probability = dropout_probability
        self.USE_BATCH_NORM = False # Wether to use batch normalization or not

        self.network = nn.Sequential()

        self.network.add_module("input_layer", nn.Linear(self.input_size, self.latent_size))
        self.network.add_module("input_dropout_layer", nn.Dropout(self.dropout_probability))
        self.network.add_module("input_relu_layer", nn.ReLU())
        if self.USE_BATCH_NORM:
            self.network.add_module("input_batchnorm_layer", nn.BatchNorm1d(self.latent_size))

        # Hidden layers
        for k in range(self.n_hidden_layers):
            self.network.add_module("hidden_layer_"+str(k), nn.Linear(self.latent_size, self.latent_size))
            self.network.add_module("hidden_dropout_layer_"+str(k), nn.Dropout(self.dropout_probability))
            self.network.add_module("hidden_relu_layer_"+str(k), nn.ReLU())
            if self.USE_BATCH_NORM:
                self.network.add_module("hidden_batchnorm_layer", nn.BatchNorm1d(self.latent_size))

        # Output layer
        self.network.add_module("output_layer", nn.Linear(self.latent_size, self.output_size))

    def forward(self, x):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            x: tensor
                The input of the netwok.

        Returns:
        --------
            The output of the network.
        """
        return self.network(x)
def create_network():
    """Create the network from the saved files.

    """

    path = str(pathlib.Path(__file__).parent.absolute()) + "/Data"
    with open(path+"/init_data.txt", 'r') as fp:
        [W0, dfree0] = json.load(fp)
    W0, dfree0 = np.array(W0), np.array(dfree0)

    with open(path+"/network_parameters.txt", 'r') as fp:
        [latent_size, n_hidden_size, output_size, input_size] = json.load(fp)

    with open(path+"/scaling.txt", 'r') as fp:
        scaling = json.load(fp)
    scaling[0] = [np.array(s) for s in scaling[0]]
    scaling[1] = [np.array(s)  for s in scaling[1]]
    scaling[2] = [np.array(s)  for s in scaling[2]]
    scaling[3] = [np.array(s)  for s in scaling[3]]

    path_model = path + "/model.pth"

    model = MLP(input_size, output_size, latent_size, n_hidden_layers=n_hidden_size, dropout_probability=0)
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    n_constraint = input_size - output_size

    return W0, dfree0, scaling, model, n_constraint
def create_data_std(X_0, mean_features_X, std_features_X):
    """Create scaled data to provide for training the model. Use mean and standard deviation for normalizing each feature.

    Parameters:
    -----------
        X_0: Tensor of numpy arrays
            Batch of X data
        mean_features_X: list of numpy array
            Mean values by X component evaluated on all the dataset.
        std_features_X: list of numpy array
            Standard deviation values by X component evaluated on all the dataset.

    Outputs:
    --------
        X: Tensor
            Scaled data ready for training MLP.
    """
    epsilon = 0.000000001  # For ensuring not dividing by 0

    X_W_0, X_dfree_0, X_s_ae = X_0
    size_W = X_W_0.shape[0]
    _mean_features_X = mean_features_X[0][np.triu_indices(n=size_W)]
    _std_features_X = std_features_X[0][np.triu_indices(n=size_W)]

    X_W_0 = X_W_0[np.triu_indices(n=size_W)]
    X = torch.cat([(X_W_0 - _mean_features_X) / (_std_features_X + epsilon),
                   (X_dfree_0 - mean_features_X[1]) / (std_features_X[1] + epsilon),
                   (X_s_ae - mean_features_X[2]) / (std_features_X[2] + epsilon)]).float()

    return X

def transform_3D_to_pos(x, y, z, angx, angy, angz):
    """
    Parameters
    ----------
    x, y, z: float
        The position of the rigid
    angx, angy, angz: float
        The orientation of the rigid (euler in degrees)

    Returns
    -------
    The position of the 3 points defining the rigid

    """
    # Compute rotation matrix xyz
    c1 = np.cos(angx * np.pi / 180)
    s1 = np.sin(angx * np.pi / 180)
    c2 = np.cos(angy * np.pi / 180)
    s2 = np.sin(angy * np.pi / 180)
    c3 = np.cos(angz * np.pi / 180)
    s3 = np.sin(angz * np.pi / 180)

    R = np.array([[c2 * c3, -c2 * s3, s2],
                  [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                  [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])

    #Coordonate of the point
    r, ang = 4, 2 * np.pi / 3
    point_coordinate = [np.matmul(R, np.array([0, r * np.sin(i * ang + np.pi / 6), r * np.cos(i * ang + np.pi / 6)])) + np.array([x, y, z]) for i in range(3)]

    return point_coordinate[0].tolist() + point_coordinate[1].tolist() + point_coordinate[2].tolist()

def predict_W_dFree(model, W0, dfree0, s_a, scaling, n_constraint, n_act_constraint, goals_pos, nb_effector = 1):

    X = [torch.tensor(W0), torch.tensor(dfree0), torch.tensor(s_a + [0]*(nb_effector*3))]
    X = create_data_std(X, scaling[0], scaling[1])

    # Prediction
    Y = model(X)

    dfree = Y[-n_constraint:].detach().numpy()
    dfree = dfree * (scaling[3][1]) + scaling[2][1]  # Rescale dfree
    dfree_a = dfree[0:n_act_constraint]
    dfree_e = dfree[n_act_constraint:]

    if nb_effector == 1:
        effector_pos_0 = [110, 0, 0]
    else:
        effector_pos_0 = transform_3D_to_pos(110, 0, 0, 0, 0, 0) #network trained for this pos
    for i in range(len(dfree_e)):
        dfree_e[i] += effector_pos_0[i] - goals_pos[i]  # (Init_pos + Relative_Disp) - Goal_pos

    W_pred = Y[:-n_constraint].detach().numpy()
    W = np.zeros((n_constraint, n_constraint))
    W[np.triu_indices(n=n_constraint)] = W_pred
    W[np.tril_indices(n=n_constraint, k=-1)] = W.T[np.tril_indices(n=n_constraint, k=-1)]
    W = W.reshape(-1)
    W = W * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1)  # Rescale W
    W = W.reshape(n_constraint, n_constraint)

    Waa = W[0:n_act_constraint, 0:n_act_constraint]
    Wea = W[n_act_constraint:, 0:n_act_constraint]

    return Waa, Wea, dfree_a, dfree_e
def init_QP(n_act_constraint, nb_add_constraints = 0):
    """
    Init QP inverse problem for a given configuration

    Parameters
    ----------
    n_act_constraint: int
        Number of actuator constraints
    nb_add_constraints: int
        By default, we only consider constraints on both actuation and actuation displacement.
        Additional constraints must be initialized.

    Outputs
    ----------
    problem: QProblem
        Initialized QProblem
    """
    nb_variables = n_act_constraint
    nb_constraints = 2 * nb_variables + nb_add_constraints
    problem = proxsuite.proxqp.sparse.QP(nb_variables, 0, nb_constraints) # number of variables, number of equality constraints, number of inequality constraints
    problem.settings.initial_guess = (proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT)
    problem.settings.max_iter = 2000

    return problem
def build_QP_system(Waa, Wea, dfree_a, dfree_e, use_epsilon=False):
    """
    Build QP inverse problem for a given configuration
    Ressource: qpoases manual at https://www.coin-or.org/qpOASES/doc/3.2/manual.pdf
               proxsuite manual at https://simple-robotics.github.io/proxsuite/md_doc_2_PROXQP_API_2_ProxQP_api.html
    ----------
    Parameters
    ----------
    problem: QProblem
        Initialized QProblem
    s_a_0: list of float
        Actuation displacement state at t=0
    s_a_t: list of float
        Actuation displacement state
    min_s_a: list of float
        Minimum relative actuation displacement state allowed
    max_s_a: list of float
        Maximum relative actuation displacement state allowed
    Waa: numpy array
        Compliance matrice projected in actuator/actuator constraint space
    Wea: numpy array
        Compliance matrice projected in effector/actuators constraint space
    dfree_a: numpy array
        Actuators displacement without any actuation
    dfree_e: numpy array
        Effectors displacement without any actuation
    use_epsilon : bool
        An energy term added in the minimization process.
        Epsilon has to be chosen sufficiently small so that the deformation energy does not disrupt the quality of the effector positioning.
    delta_a_var: float
        Maximum actuator displacement variation allowed between two dt simulation steps
    ----------
    Outputs
    ----------
    H: numpy array
        Quadratic cost matrix
    g: numpy array
        Linear cost matrix
    A: numpy array
        Constraint matrix
    lb: numpy array
        Lower bound for actuation
    ub: numpy array
        Upper bound for actuation
    lbA: numpy array
        Lower bound for actuation displacement
    ubA: numpy array
        Upper bound for actuation displacement
    """

    # Init QP problem with our data at dt
    if use_epsilon:
        epsilon = 0.01 * np.linalg.norm(np.dot(np.transpose(Wea), Wea), ord=1) / np.linalg.norm(Waa, ord=1)
    else:
        epsilon = 0.0

    H = np.dot(np.transpose(Wea), Wea).astype('double') + epsilon * (Waa).astype('double')
    g = np.dot(np.transpose(Wea), dfree_e).astype('double')
    A = Waa.astype('double')
    lb = np.array([0 for i in range(len(dfree_a))]).astype(
        'double')  # Lower bound for actuation effort constraint - specific to a cable
    ub = np.array([np.inf for i in range(len(dfree_a))]).astype(
        'double')  # Upper bound for actuation effort constraint - specific to a cable

    lbA = (np.array([-np.inf for i in range(len(dfree_a))])).astype('double')
    ubA = (np.array([np.inf for i in range(len(dfree_a))])).astype('double')

    return H, g, A, lb, ub, lbA, ubA
def solve_QP(problem, H, g, A, lb, ub, lbA, ubA, is_init = True):
    """
    Solve QP inverse for a given configuration
    Ressource: qpoases manual at https://www.coin-or.org/qpOASES/doc/3.2/manual.pdf
               proxsuite manual at https://simple-robotics.github.io/proxsuite/md_doc_2_PROXQP_API_2_ProxQP_api.html
    ----------
    Parameters
    ----------
    problem: QProblem
        Initialized QProblem
    H: numpy array
        Quadratic cost matrix
    g: numpy array
        Linear cost matrix
    A: numpy array
        Constraint matrix
    lb: numpy array
        Lower bound for actuation
    ub: numpy array
        Upper bound for actuation
    lbA: numpy array
        Lower bound for actuation displacement
    ubA: numpy array
        Upper bound for actuation displacement
    is_init: bool
        True if the solver has already been initialized
    ----------
    Outputs
    ----------
    lambda_a: numpy array
        Actuation displacement
    """
    nWSR = 2000 # Number of QP iterations
    n_constraints = len(lb)


    """
    QP problem shape is:
    argmin_x  1/2 xT H x + gT(w0) x
    s. t. A(w0)x ≤ b(w0) ,
    C(w0)x ≤ u(w0)
    """

    # Displacement constraints
    C = A
    l = lbA
    u = ubA

    # Actuation effort constraints
    # Constraint lb(w0) ≤ x ≤ ub(w0) is equivalent to constraints -x ≤ -lb(w0) and x ≤ ub(w0)
    nb_variables = H.shape[0]
    I = np.eye(nb_variables)

    # Final constraint matrices
    C = np.concatenate((C, np.array(I)), axis=0)
    l = np.concatenate([l, lb])
    u = np.concatenate([u, ub])

    if is_init:
        problem.update(H, g, None, None, C, l, u)
    else:
        problem.init(H, g, None, None, C, l, u)

    problem.solve()
    lambda_a = problem.results.x

    return lambda_a


def compute_delta_a(lambda_a, Waa, dfree_a):
    return np.matmul(Waa, lambda_a) + dfree_a
def current_actuation():
    print("[ERROR] >> Recover the true value of the actuation (volumeGrowth of each cavity).")
    exit(1)
def send_to_robot(lambda_a):
    print("[ERROR] >> Apply lambda_a to your robot.")
    exit(1)
def current_effector_pos():
    print("[ERROR] >> Recover the true value of the effector position.")
    exit(1)