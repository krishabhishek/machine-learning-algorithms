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

* Run Gaussian Mixture Model processor
```
/usr/bin/python3 "$CODEDIR"/gmm_classification.py --input_data_folder "$DATADIR"
```

* Run Hidden Markov Model processor
```
/usr/bin/python3 "$CODEDIR"/hmm_monitoring.py --input_data_folder "$DATADIR"
```
