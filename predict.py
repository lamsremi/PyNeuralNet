"""
Script for prediction.
"""
import pickle
import importlib

import numpy as np
import pandas as pd

import tools


@tools.debug
def main(data_df,
         data_source,
         model_type,
         model_version):
    """
    Main prediction function.
    Args:
        data_df (DataFrame): input to predict.
        data_source (str): source of the data.
        model_type (str): type of the chosen model for the prediction.
        model_version (str): version of model to use.
    Return:
        prediction (ndarray)
    """
    # Load normalized parameters
    norm_params = load_params(data_source)

    # Process input
    features = pre_process(
        features=np.array(data_df),
        norm_params=norm_params)

    # Init a model if none
    model = init_model(
        model_type=model_type)

    # Load the model parameters TODO
    model.load_parameters(model_version=model_version)

    # Predict TODO
    prediction = model.predict(features)

    # Post process the prediction
    prediction = post_process(
        prediction=prediction,
        norm_params=norm_params)

    # Return value
    return prediction


# @tools.debug
def load_params(data_source):
    """Load normalized parameters."""
    with open("data/{}/norm_params.pkl".format(data_source), 'rb') as handle:
        norm_params = pickle.load(handle)
    return norm_params


def pre_process(features, norm_params):
    """Preprocess."""
    scaled_features, features_mean, features_std = scale(
        data=features,
        data_mean=norm_params["features"]["mean"],
        data_std=norm_params["features"]["std"])
    return scaled_features


def post_process(prediction, norm_params):
    """Postprocess."""
    unscaled_prediction = unscale(
        scaled_data=prediction,
        data_mean=norm_params["truth"]["mean"],
        data_std=norm_params["truth"]["std"])
    return unscaled_prediction


def scale(data, data_mean=None, data_std=None):
    """Scale the data

    Args:
        data (ndarray): data
        data_mean (float): mean of the data.
        data_std (float): standard deviation of the data.
    """
    # Compute mean if not given
    if data_mean is None:
        data_mean = np.mean(data, axis=0)
    # Compute standard deviation if not given
    if data_std is None:
        data_std = np.std(data, axis=0)
    # Perform the scaling
    scaled_data = (data - data_mean) / data_std
    return scaled_data, data_mean, data_std

def unscale(scaled_data, data_mean, data_std):
    """Unscale data
    """
    # Perform the scaling
    data = scaled_data*data_std + data_mean
    return data


def init_model(model_type):
    """
    Init a model.
    Args:
        model_type (str): type of the model to init.
    Return:
        model (object): loaded model
    """
    # Import the good model
    model_class = importlib.import_module("library.{}.model".format(model_type))
    # Init the instance
    model = model_class.Model()
    # Return value
    return model


if __name__ == '__main__':
    input_df = pd.read_pickle("data/health/data.pkl").iloc[100:101, :-1].copy()
    for source in ["health"]:
        for model_str in ["doityourself", "keras"]:
            main(
                data_df=input_df,
                data_source=source,
                model_type=model_str,
                model_version=source)
