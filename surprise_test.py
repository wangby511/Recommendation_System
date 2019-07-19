from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')

data.split(n_folds=3)

algo = SVD()

# perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
#
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
# RMSE (testset)    0.9354  0.9342  0.9387  0.9358  0.9340  0.9356  0.0017
# MAE (testset)     0.7381  0.7361  0.7408  0.7373  0.7372  0.7379  0.0016
# Fit time          5.66    5.18    5.64    5.59    5.63    5.54    0.18
# Test time         0.20    0.21    0.19    0.19    0.20    0.20    0.01