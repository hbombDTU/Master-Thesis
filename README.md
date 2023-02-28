# Master-Thesis

## Descripton:
This repository contains codes that have been used in my Master Thesis: Using Curtailed Power Measurements to Improve Probabilistic Forecasts

### Code:
```data_pre.py``` Preprares the data that is to be used for training the model. The script preprocesses and scales the input matrix needed for the regression and GMM models.

```Tobit.py``` Models the data using Tobit regression. 

```JFST.py``` Models the data using JFST regression.

```GMM.py``` Prepares the data to be used for the Gaussain Mixture Models (GMM). It also computes the baseline given in the paper.

### Test:
```test_clustering.py``` Tests the ```GMM.py``` script

```test_regression.py``` Tests the ```Tobit.py``` and ```JFST.py``` script
