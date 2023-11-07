# -*- coding: utf-8 -*-
"""Main file to launch script (learning, data acquisition, applications).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 05 2022"

# System libs
from copyreg import pickle
import pathlib
import pickle

# Local libs
import peewee

########################################################################################
############################# Database definition ######################################
########################################################################################
database_path = pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + "/../Results/data.db")
db = peewee.SqliteDatabase(database_path, pragmas={
                                'journal_mode': 'wal', # Allow both writer and readers to co-exist
                                'cache_size': -1024 * 64,# Page cache size
                                "foreign_keys": 1, # Enforce foreign key constraint
                                "ignore_check_constraints": 0, # Check constraints
                                })

class BaseModel(peewee.Model):
        class Meta:
            database = db

class Models(BaseModel):
    """ Table for models

    Attributes
    ----------
    id : int
        Primary key
    name: string
        Name of the model
    data: timestamp
        Date when the model is inserted in the database
    """
    id = peewee.AutoField(primary_key = True)
    name = peewee.CharField(unique=True)
    date = peewee.DateTimeField(constraints=[peewee.SQL('DEFAULT CURRENT_TIMESTAMP')]) # Creation date

class SimulationData(BaseModel):
    """ Table for simulation results

    Attributes
    ----------
    id : int
        Primary key
    id_model: int
        Reference toward the corresponding model
    id_sampling_stats: int
        Reference towards the corresponding sampling stats
    design_params: pickle list
        Value for each design variable
    W_0: pickle numpy array
        Compliance matrice projected in constraint space without constraint
    dfree_0: pickle numpy array
        Displacement in free configuration without any actuation
    a: pickle list
        Actuations displacements
    s_a: pickle list
        Actuation displacement state
    s_e: pickle list
        Effector displacement state
    W: pickle numpy array
        Compliance matrice projected in constraint space for given actuation a
    dfree: pickle numpy array
        Displacement in free configuration without any actuation for a given configuration
    is_train: boolean
        Check if simulation data is from train or test set
    """
    id = peewee.AutoField(primary_key = True)
    id_model = peewee.IntegerField()
    id_sampling_stats = peewee.IntegerField()
    design_params = peewee.BlobField()
    W_0 = peewee.BlobField()
    dfree_0 = peewee.BlobField()
    a = peewee.BlobField()
    s_a = peewee.BlobField()
    s_e = peewee.BlobField()
    W = peewee.BlobField()
    dfree = peewee.BlobField()
    is_train = peewee.BooleanField()

    class Meta:
        constraints = [peewee.SQL('FOREIGN KEY(id_model) REFERENCES models(id)'),
        peewee.SQL('FOREIGN KEY(id_sampling_stats) REFERENCES samplingstats(id)')]


class SamplingStats(BaseModel):
    """ Table for managing sampling stats during Data Acquisition

    Attributes
    ----------
    id: int
        Primary key
    id_model : str
        Reference toward the corresponding model
    sampling_strategy: string
        Name of the sampling strategy
    n_samples: int
        Number of samples for sampn_samplesling strategy
    n_curr_sample: int
        Number of samples already evaluated
    """

    id = peewee.AutoField(primary_key = True)
    id_model = peewee.IntegerField()
    sampling_strategy = peewee.CharField()
    n_samples = peewee.IntegerField()
    n_curr_sample = peewee.IntegerField()

    class Meta:
        constraints = [peewee.SQL('FOREIGN KEY(id_model) '
                           'REFERENCES models(id)'),
            peewee.SQL('UNIQUE (id_model, sampling_strategy, n_samples)')]


class LearningStats(BaseModel):
    """ Table for managing learning stats

    Attributes
    ----------
    id: int
        Primary key
    id_model : int
        Reference toward the corresponding model
    id_sampling: int
        Reference towards the corresponding sampling strategy
    network_name: str 
        The name of the network
    mode_loss: str 
        The type of the loss we want to use
    n_hidden_layers: int
        Number of hidden layers in the model
    latent_size: int
        Latent size for the hidden size of the newtork
    batch_size: int
        Batch_size for the learning
    data_normalization: str
        The adopted data normalization strategy
    normalization_vector: pickle numpy array
        The normalization vector applied to input data before training the MLP.
        Depends of the adopted data normalization strategy.
    """

    id = peewee.AutoField(primary_key=True)
    id_model = peewee.IntegerField()
    id_sampling = peewee.IntegerField()
    network_name = peewee.CharField()
    mode_loss = peewee.CharField()
    n_hidden_layers = peewee.IntegerField()
    latent_size = peewee.IntegerField()
    batch_size = peewee.IntegerField()
    data_normalization = peewee.CharField()
    normalization_vector = peewee.BlobField()

    class Meta:
        constraints = [peewee.SQL('FOREIGN KEY(id_model) '
                                  'REFERENCES models(id)'), 
                        peewee.SQL('FOREIGN KEY(id_sampling) '
                                  'REFERENCES samplingStats(id)'),           
                       peewee.SQL('UNIQUE (id_model, id_sampling, network_name, mode_loss, n_hidden_layers, latent_size, batch_size, data_normalization)')]


def create_tables():
    with db:
        db.create_tables([Models, SimulationData,
                         SamplingStats, LearningStats])

def connect_db():
    db.connect()

def disconnect_db():
    db.close()

########################################################################################
############################# Creating new objects #####################################
########################################################################################
def add_model(model_name):
    """ Add a new model in the Model table

    Parameters
    ----------
    model_name : string
        Name of the model
    """
    try:
        Models.create(name = model_name)
    except peewee.IntegrityError:
        print("Model name " + model_name + " already registered in database")

def add_simulation_results(model_name, id_sampling_stats, design_params, W_0, dfree_0, a, s_a, s_e, W, dfree, is_train):
    """ Add new simulation results in the SimulationData table

    Parameters
    ----------
    model_name : string
        Name of the model
    id_sampling_stats: int
        Reference towards the corresponding sampling stats
    design_params: pickle list
        Value for each design variable
    W_0: pickle numpy array
        Compliance matrice projected in constraint space without constraint
    dfree_0: pickle numpy array
        Displacement in free configuration without any actuation
    a: pickle list
        Actuations displacements
    s_a: pickle list
        Actuation displacement state
    s_e: pickle list
        Effector displacement state
    W: pickle numpy array
        Compliance matrice projected in constraint space for given actuation a
    dfree: pickle numpy array
        Displacement in free configuration without any actuation for a given configuration
    is_train: boolean
        Check if simulation data is from train or test set
    """
    try:
        id_model = Models.select(Models.id).where(Models.name == model_name).get()
        SimulationData.create(id_model = id_model, id_sampling_stats = id_sampling_stats, design_params = design_params, W_0 = W_0, dfree_0 = dfree_0, a = a, s_a = s_a, s_e = s_e, W = W, dfree = dfree, is_train = is_train)
    except peewee.IntegrityError as e:
        print("Integrity error is:", e)

def add_sampling_stats(model_name, sampling_strategy, n_samples):
    """ Add a new sampling strategy

    Parameters
    ----------
    id: int
        Primary key
    id_model : str
        Reference toward the corresponding model
    sampling_strategy: string
        Name of the sampling strategy
    n_samples: int
        Number of samples for sampling strategy

    Outputs
    ----------
    id: int
        Id of the newly inserted sampling stat line
    """
    try:
        id_model = Models.select(Models.id).where(Models.name == model_name).get()
        sampling_stats = SamplingStats.create(id_model = id_model, sampling_strategy = sampling_strategy, n_samples = n_samples, n_curr_sample = -1)
        return sampling_stats.id
    except peewee.IntegrityError as e:
        print("Integrity error is:", e)

def add_learning_stats(model_name, sampling_strategy, n_samples, network_name, mode_loss, n_hidden_layers, latent_size, batch_size, data_normalization, normalization_vector = []):
    """ Add a new configuration of the network
    ----------
    Parameters
    ----------
    model_name : str
        The name of the robot
    sampling_strategy: string
        Name of the sampling strategy
    n_samples: int
        Number of samples 
    network_name: str 
        The name of the network
    mode_loss: str 
        The type of the loss we want to use
    n_hidden_layers: int
        Number of hidden layers in the model
    latent_size: int
        Latent size for the hidden size of the newtork
    batch_size: int
        Batch_size for the learning
    data_normalization: str:
        The adopted data normalization strategy
    normalization_vector: pickle numpy array
        The normalization vector applied to input data before training the MLP.
        Depends of the adopted data normalization strategy.
    ----------
    Outputs
    ----------
    id: int
        Id of the newly inserted sampling stat line
    """
    try:
        id_model = Models.select(Models.id).where(Models.name == model_name).get()
        id_sampling = SamplingStats.select(SamplingStats.id).where(SamplingStats.id_model == id_model, SamplingStats.sampling_strategy == sampling_strategy, SamplingStats.n_samples == n_samples).get()
        sampling_stats = LearningStats.create(
            id_model=id_model, id_sampling=id_sampling, network_name=network_name, mode_loss=mode_loss, n_hidden_layers=n_hidden_layers, latent_size=latent_size, batch_size=batch_size, data_normalization = data_normalization, normalization_vector = pickle.dumps(normalization_vector))
        return sampling_stats.id
    except peewee.IntegrityError as e:
        print("Integrity error is:", e)


########################################################################################
################################ Update database #######################################
########################################################################################
def update_sampling_stats(model_name, sampling_strategy, n_samples):
    model = Models.get(Models.name == model_name)
    q = SamplingStats.update({SamplingStats.n_curr_sample: SamplingStats.n_curr_sample + 1}).where(SamplingStats.id_model == model.id, SamplingStats.sampling_strategy == sampling_strategy, SamplingStats.n_samples == n_samples)
    q.execute()

########################################################################################
################################ Query database ########################################
########################################################################################
def query_sampling_stats(model_name, sampling_strategy, n_samples):
    model = Models.get(Models.name == model_name)
    query = SamplingStats.select().where(SamplingStats.id_model == model.id, SamplingStats.sampling_strategy == sampling_strategy, SamplingStats.n_samples == n_samples).dicts()
    return query

def query_sampling_stats_from_id(id):
    query = SamplingStats.select().where(SamplingStats.id == id).dicts()
    return query

def query_sampling_stats_for_a_model(model_name):
    model = Models.get(Models.name == model_name)
    query = SamplingStats.select().where(SamplingStats.id_model == model.id).dicts()
    return query

def query_learning_stats(model_name, sampling_strategy, n_samples, network_name, mode_loss, n_hidden_layers, latent_size, batch_size, data_normalization):
    id_model = Models.select(Models.id).where(Models.name == model_name).get()
    id_sampling = SamplingStats.select(SamplingStats.id).where(SamplingStats.id_model == id_model, SamplingStats.sampling_strategy == sampling_strategy, SamplingStats.n_samples == n_samples).get()
    query = LearningStats.select().where(LearningStats.id_model == id_model, LearningStats.id_sampling == id_sampling,
                                         LearningStats.network_name == network_name, LearningStats.mode_loss == mode_loss, LearningStats.n_hidden_layers == n_hidden_layers, LearningStats.latent_size == latent_size, LearningStats.batch_size == batch_size, LearningStats.data_normalization == data_normalization).dicts()
    return query

def query_learning_stats_for_a_SS(model_name, network_name, mode_loss, n_hidden_layers, latent_size, batch_size, data_normalization):
    id_model = Models.select(Models.id).where(Models.name == model_name).get()
    query = LearningStats.select().where(LearningStats.id_model == id_model, 
                                         LearningStats.network_name == network_name, LearningStats.mode_loss == mode_loss, LearningStats.n_hidden_layers == n_hidden_layers, LearningStats.latent_size == latent_size, LearningStats.batch_size == batch_size, LearningStats.data_normalization == data_normalization).dicts()
    return query

def query_learning_stats_for_a_NN(model_name, network_name):
    model = Models.get(Models.name == model_name)
    query = LearningStats.select().where(LearningStats.id_model == model.id, LearningStats.network_name == network_name).dicts()
    return query

def query_sampling_strategy_from_model(model_name):
    model = Models.get(Models.name == model_name)
    query = SamplingStats.select(SamplingStats.sampling_strategy).where(SamplingStats.id_model == model.id).dicts()
    return query

def query_n_samples_from_model(model_name, sampling_strategy):
    model = Models.get(Models.name == model_name)
    query = SamplingStats.select(SamplingStats.n_samples).where(SamplingStats.id_model == model.id, SamplingStats.sampling_strategy == sampling_strategy).dicts()
    return query


def query_id_sample_from_model(model_name, sampling_strategy, n_samples):
    model = Models.get(Models.name == model_name)
    query = SamplingStats.select(SamplingStats.id).where(
        SamplingStats.id_model == model.id, SamplingStats.sampling_strategy == sampling_strategy, SamplingStats.n_samples == n_samples).dicts()
    return query

def query_simulation_data(model_name, id_sampling_stats, is_train):
    model = Models.get(Models.name == model_name)
    query = SimulationData.select(SimulationData.design_params, SimulationData.W_0, SimulationData.dfree_0, SimulationData.s_a, SimulationData.s_e, SimulationData.W, SimulationData.dfree).where(SimulationData.id_model == model.id, SimulationData.id_sampling_stats == id_sampling_stats, SimulationData.is_train == is_train).dicts()
    return query

def query_recover_W_dfree_from_s_a(model_name, id_sampling_stats, is_train, s_a):
    model = Models.get(Models.name == model_name)
    query = SimulationData.select(SimulationData.W, SimulationData.dfree).where(SimulationData.id_model == model.id, SimulationData.id_sampling_stats == id_sampling_stats, SimulationData.is_train == is_train, SimulationData.s_a.in_(s_a)).dicts()
    return query

