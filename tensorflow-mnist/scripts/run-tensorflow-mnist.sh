#!/usr/bin/env bash

CODEDIR=$(dirname "$0")"/../"

# Run Gaussian Mixture Model processor
/usr/bin/python3 "$CODEDIR"/tensorflow_mnist.py
