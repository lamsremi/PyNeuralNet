# Neural Networks Implementations


## Introduction

This module presents 3 implementations of the same neural network:
* Using the high level framework Keras
* Using the lower level framework TensorFlow
* From scratch using only numpy library

The implemented model is a basic neural network with the following attributes:
* 2 layers made of 10 and 15 neurons
* 2 sigmoid activations
* Stochastic gradient descent with a learning rate of 0.005
* Mean absolute error as cost function


## Motivation

The purpose of this exercice is to gain a complete understanding on how neural networks work (forward and backward propagation), which helps to understand and design more advanced deep learning models such as RNN or encoder-decoder.

It is also a valuable exercice to practice and improve python programming skills.


## Code structure

The code is structured as follow:
```
pyNeuralNet
├- data/
│   └- health/
│       ├- raw_data/
│       │   └- data.csv
│       └- data.pkl
├- library/
│   ├- doityourself/
│   │   ├- params/
│   │   └- model.py
│   ├- keras/
│   │   ├- params/
│   │   └- model.py
│   └- tensorflow/
│       ├- params/
│       └- model.py
├- unittest/
│   └- test_core.py
├- .gitignore
├- predict.py
├- README.md
├- requirements.txt
├- tools.py
└- train.py
```

## Installation

To dowload the different implementations of neural networks, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyNeuralNet.git
```

Then install the requirements in your local environment or in a virtual one that was previously created :

```
$ pip install -r requirements.txt
```

## Test

To test if all the functionnalities are working :

```
$ python -m unittest discover -s unittest
```

## Use

For training :

```
>>> from train import main
>>> for source in ["health"]:
        for model in ["doityourself", "keras", "tensorflow]:
            main(data_df=None,
                 data_source=source,
                 model_type=model,
                 model_version=source)
```

For predicting :

```
>>>from predict import main
>>> input_df = pd.read_pickle("data/health/data.pkl").iloc[100:101, :-1].copy()
    for source in ["health"]:
        for model_str in ["doityourself", "keras", "tensorflow"]:
            main(
                input_df,
                data_source=source,
                model_type=model_str,
                model_version=source)
```


## Author

Rémi Moise

moise.remi@gmail.com

## License

MIT License

Copyright (c) 2017 Rémi Moïse