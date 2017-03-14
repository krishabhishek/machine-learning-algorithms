#!/usr/bin/env bash

CODEDIR=$(dirname "$0")"/../"
DATADIR=$(dirname "$0")"/../data/"

# Run Gaussian Mixture Model processor
/usr/bin/python3 "$CODEDIR"/gmm_classification.py --input_data_folder "$DATADIR"

# Run Hidden Markov Model processor - Forward Algorithm
/usr/bin/python3 "$CODEDIR"/hmm_monitoring.py --input_data_folder "$DATADIR"

# Run Hidden Markov Model processor - Viterbi Algorithm
/usr/bin/python3 "$CODEDIR"/hmm_monitoring.py --input_data_folder "$DATADIR" --viterbi
