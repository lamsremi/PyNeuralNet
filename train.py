"""
Script to train models.
"""
import pickle
import importlib

import pandas as pd
import numpy as np

import tools


pd.set_option('display.width', 300)

def main(data_df=None,
         data_source=None,
         model_type=None,
         model_version=None):
    """
    Main function for training.
    """
    if data_df is None:
        # Load labaled data
        data_df = load_labaled_data(data_source)

    # Divide into features and truth
    features, truth, norm_params = process_data(
        features=np.array(data_df.iloc[:, :-1]),
        truth=np.array(data_df.iloc[:, -1:])
    )

    # Store normalized parameters
    store_params(norm_params, data_source)

    # Init the model
    model = init_model(
        model_type=model_type)

    # Train the model
    model.fit(features, truth,
              input_dim=features.shape[-1], output_dim=truth.shape[-1],
              batch_size=50, epochs=5)

    # # Store the model parameters.
    # model.persist_parameters(model_version=model_version)

    # # Return
    # return True


# @tools.debug
def load_labaled_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle("data/{}/data.pkl".format(data_source))
    return data_df


# @tools.debug
def process_data(features, truth):
    """Process the data

    Process th data in the right format.

    Args:
        features (nd_array): input.
        truth (nd_array): truth.

    Return:
        scaled_features (ndarray): input data.
        scaled_truth (ndarray): expected output data.
        norm_params (dict): normalized parameters.
    """
    # Scale the data
    scaled_features, features_mean, features_std = scale(features)
    scaled_truth, truth_mean, truth_std = scale(truth)

    # Combine
    norm_params = {
        "features": {
            "mean": features_mean,
            "std": features_std
        },
        "truth": {
            "mean": truth_mean,
            "std": truth_std
        }
    }
    return scaled_features, scaled_truth, norm_params


def store_params(norm_params, data_source):
    """Store normalized parameters."""
    with open("data/{}/norm_params.pkl".format(data_source), 'wb') as handle:
        pickle.dump(norm_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def unscale(np_scaled_data, data_mean, data_std):
    """Unscale data
    """
    np_data = np_scaled_data*data_std + data_mean
    return np_data


def init_model(model_type):
    """
    Init a model.
    Args:
        model_type (str): type of the model to init.
    Return:
        model (object): loaded model
    """
    # Import the library
    model_class = importlib.import_module("library.{}.model".format(model_type))
    # Inits the model instance
    model = model_class.Model()
    # Return value
    return model


if __name__ == '__main__':
    for source in ["health"]:
        for model_str in ["tensorflow", "keras", "doityourself"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model_str,
                 model_version=source)
