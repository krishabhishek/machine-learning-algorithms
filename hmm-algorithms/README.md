# HMM Monitoring vs. Mixtures of Gaussians

## Pre-requisites
* Numpy

## Run Commands

### Comparing Hidden Markov models to Gaussian Mixture Models

* Set environment variables
```
export CODEDIR=<CODE_DIRECTORY>
export DATADIR=<DATA_DIRECTORY>
```

* Run Gaussian Mixture Model classifier
```
/usr/bin/python3 "$CODEDIR"/gmm_classification.py --input_data_folder "$DATADIR"
```

* Run Hidden Markov Model - Forward Algorithm - Monitoring
```
/usr/bin/python3 "$CODEDIR"/hmm_classification.py --input_data_folder "$DATADIR"
```

* Run Hidden Markov Model - Viterbi Algorithm - Sequence Prediction
```
/usr/bin/python3 "$CODEDIR"/hmm_classification.py --input_data_folder "$DATADIR" --viterbi
```
