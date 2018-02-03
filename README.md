# Neural networks implementations


## Synopsys

This project aims to recode the algorithms for using and training neural networks.

## Motivation

The purposes are to learn how does a neural network work and how it is being trained and improve programming skills.

## Code structure

The "doityourself" model is compared to an keras based implementation using the sa,e hyper parameters.

## Pseudocode

```
# Training
>>> for source in ["health"]:
        for model_str in ["doityourself", "keras"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model_str,
                 model_version=source)

# Prediction
>>> input_df = pd.read_pickle("data/health/data.pkl").iloc[100:101, :-1].copy()
    for source in ["health"]:
        for model_str in ["doityourself", "keras"]:
            main(
                input_df,
                data_source=source,
                model_type=model_str,
                model_version=source)
```

## Installation

To dowload the different implementations of neural networks, you can directly clone the repository

```
git clone https://github.com/lamsremi/pyNeuralNet.git
```

## Author

Rémi Moise

## TODO
