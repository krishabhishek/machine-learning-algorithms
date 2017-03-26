#!/usr/bin/env bash

CODEDIR=$(dirname "$0")"/../"

# Run Softmax network processor
/usr/bin/python3 "$CODEDIR"/tensorflow_mnist.py --mode softmax

# Run Convolutional Neural Network learning processor
/usr/bin/python3 "$CODEDIR"/tensorflow_mnist.py --mode cnn
