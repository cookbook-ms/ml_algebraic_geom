# Machine learning detects terminal singularities

This repository is the official implementation of the code used in *Machine learning detects terminal singularities* by Tom Coates, Alexander Kasprzyk, and Sara Veneziale.

## Requirements

The environment requirements are found in `requirements.txt`. To set up the conda environment run 

```setup
conda create --name <env> --file requirements.txt
```

The scripts have been executed in `Python v3.10.8`, using the following packages `scikit-learn v1.1.3`, `pytorch v1.13.1`, `matplotlib v3.7.1`, `numpy v1.24.2`, `seaborn v0.12.2`, `ray v2.4.0`. 

Part of the data generation steps for the dataset of 100M probably-$\mathbb{Q}$-Fano varieties uses `SageMath v9.8`.

## Outputs

All images are saved in the folders `images_{DPI}` for $\texttt{DPI} = 300, 600, 1200$ respectively.

All trained models (scalers and neural network weights) are saved as `trained_models/ml_terminality_scaler_{TRAIN_SIZE}_dim8_bound{BOUND}.pkl` and `trained_models/ml_terminality_nn_{TRAIN_SIZE}_dim8_bound{BOUND}.pt`, for each value of `TRAIN_SIZE` and `BOUND`.

## Data

The full dataset is available on Zenodo doi:10.5281/zenodo.10046893

It should be downloaded and extracted to `data/`.
## Training

To train the model, run the script `code/training_script_terminality.py`. The chosen bound on the weights is hard-coded in the script.

```train
python training_script_terminality.py TRAIN_SIZE
```

where `TRAIN_SIZE` is the desired training size.

`TRAIN_SIZE = 2500000` should produce the same accuracy as the final neural network used in the main text. Different `TRAIN_SIZE` values are used to produce the accuracy results plotted in the learning curve in the main text.

The training uses hyperparameter configuration
```python
    config = {
        "layers": (512,768,512),
        "lr": 0.01,
        "batch_size": 128,
        "momentum": 0.99, 
        "slope": 0.01
        }
```

Note that the script calls functions from `code/training_functions.py`, which is in the same folder.

The script saves the standard scaler as a pickle file `trained_models/ml_terminality_scaler_{TRAIN_SIZE}_dim8_bound{BOUND}.pkl` and the best neural network weights in `trained_models/ml_terminality_nn_{TRAIN_SIZE}_dim8_bound{BOUND}.pt`.

The script will produce a learning curve image `learning_curve_terminality_{TRAIN_SIZE}_dim8_bound{BOUND}.png`. The script saves the data necessary to plot the learning curves in text files `losses/loss_train_{TRAIN_SIZE}_bound{BOUND}.txt` and `losses/loss_validation_{TRAIN_SIZE}_bound{BOUND}.txt`.

## Evaluation

To evaluate the model trained on a specific `TRAIN_SIZE` run the script `code/testing_script_terminality.py`. The chosen bound on the weights is hard-coded in the script.

```eval
python testing_script_terminality.py TRAIN_SIZE
```

It will print both training and testing accuracy. Note that the script calls functions from `code/training_functions.py`, which is in the same folder.

The file will produce confusion matrices and save them as `images_{DPI}/confusion_matrix_terminality_{TRAIN_SIZE}_true_dim8_bound{BOUND}.png` and `images_{DPI}/confusion_matrix_terminality_{TRAIN_SIZE}_pred_dim8_bound{BOUND}.png`.

## Trained models

The trained models used for the results in the paper are available in the folder `trained_models_final`.

## Hyperparameter tuning

The hyperparameter tuning is carried out in part using Ray Tune using the script `code/tuning_script.txt`. The chosen bound on the weights is hard-coded in the script.

```tune
python tuning_script.txt /path/to/data
```
where `/path/to/data` is the path to the directory where the terminal and non-terminal datasets are saved.

Because of memory issues, it is better to have separate smaller `.txt` files containing the data used for tuning called `terminal_tuning_bound{BOUND}.txt` and `non_terminal_tuning_bound{BOUND}.txt`.

## Data generation

To generate the dataset of $100$ million probably-$\mathbb{Q}$-Fano toric varieties of Picard rank two and dimension eight from Section 6 of the paper, we run the python script `code/random_script_hpc.py` on the HPC cluster in parallel over 1200 cores.

```eval
python testing_script_terminality.py TRAIN_SIZE
```
Note that these scripts require Sage Integers to run, so they need to be run in a Python environment that has `SageMath v9.8` installed.

The script writes to `.txt` files in the folder `data_hpc/` named by the job-id (which is hardcoded in the file, but on the HPC should be read from the environment). The lines of each output file correspond to probably-$\mathbb{Q}$-Fano varieties and each line consists of a dictionary of the following form:

```python
{
    ULID: '01H0MFY6DXJSTESCNZ5GTZJXYD',
    Weights: '[[1, 6, 5, 5, 3, 5, 3, 5, 0, 0], [0, 2, 2, 3, 2, 6, 4, 7, 3, 5]]',
    Probability: 0.9999924898147583,
    Regression: '[2.1742922271966085, -1.8413496900142725]',
    FanoIndex: 1,
    K: '[33, 34]',
    Alpha: 1.2160491943359375
}
```
We explain the meaning of the keys here:

- `ULID`: alphanumeric string that identifies each sample.

- `Weights`: a string of the form `'[[a1, a2, ..., a10],[b1, b2, ...,b10]]'` where a1, ... , a10, b1, ... , b10 are non-negative integers.

- `Probability`: a float representing how confidently the neural network classifies the example as terminal.

- `Regression`: a string of the form `'[A,B]'` where A and B are the two coefficients in the asymptotic expression for the period sequence.

- `FanoIndex`: an integer, the greatest common divisor of the sum of the ai's and the sum of the bi's.

- `K`: a string of the form `'[a,b]'` where a is the sum of the ai's and b is the sum of the bi's.

- `Alpha`: the float $\nu$, such that $(1, \nu)$ is a solution to the appropriate homogeneous polynomial equation (see Section 6 in the main text).

The data is combined and the entries are deduplicated using the python script `code/deduplicating_100M.py`. The output of this script is a text file `data/terminal_dim8_probable.txt` which looks like this.

```
Weights: [[1, 5, 5, 5, 5, 4, 3, 5, 2, 0], [0, 0, 5, 5, 6, 5, 4, 7, 4, 4]]
Regression: [2.1866153690022787, -0.37795463861961665]
FanoIndex: 5

Weights: [[4, 5, 7, 7, 5, 7, 6, 2, 1, 0], [0, 0, 2, 2, 2, 3, 5, 3, 5, 6]]
Regression: [2.2532218607841252, -1.4715747146469598]
FanoIndex: 4
```

## Images

Here we associate scripts to the corresponding images in the paper. 

| Figure        | Script                                   |
| --------------|------------------------------------------|
| Fig. 1        | `code/images.nb` (Mathematica notebook) |
| Fig. 2a       | `code/learning_curve_script.py`         |
| Fig. 2b       | `code/training_script_terminality.py`   |
| Fig. 3a       | `code/landscape_fano_index.py`          |
| Fig. 3b       | `code/landscape_frequency.py`           |
| Fig. 4        | `code/limitations_script.py`            |
| Supp Fig. 1   | `code/testing_script_terminality.py`    |
| Supp Fig. 2   | `code/products_wps.py`                  |
| Supp Fig. 3   | `code/smooth_landscape.py`              |
| Supp Fig. 4   | `code/histogram.py`                     |
| Supp Fig. 5   | `code/overlapping.py`                   |
| Supp Fig. 6a  | `code/learning_curve_script.py`         |
| Supp Fig. 6b  | `code/training_script_terminality.py`   |
| Supp Fig. 6c  | `code/training_script_terminality.py`   |
| Supp Fig. 7   | `code/limitations_script.py`            |


