import utils.utility as ut
from utils.feature_creator import FeatureCreator

def create_features():
    dae_params      = ut.load_dae_config("config/cnf_dae.csv")
    data            = ut.load_raw_data("data/origin_data/", dae_params["nclasses"])
    data_processor  = FeatureCreator()
    features        = data_processor.get_features(data, dae_params)
    ut.save_data(features, dae_params["p_training"])
