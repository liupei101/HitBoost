# coding=utf-8
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb

from hitcore import *
from hitboost import hitboost

def parse_data(data, T_col, E_col, exclude_col=[]):
    x_cols = [c for c in data.columns if c not in [T_col, E_col] + exclude_col]
    data.loc[:, 'Y'] = data.loc[:, T_col]
    # Negtive values are considered right censored
    data.loc[data[E_col] == 0, 'Y'] = - data.loc[data[E_col] == 0, 'Y']
    # returns X, Y
    return data[x_cols], data['Y'].values.astype('int')

def load_data(filename, T_col, E_col, exclude_col=[]):
    data = pd.read_csv(filename)
    return parse_data(data, T_col, E_col, exclude_col=exclude_col)

############################## Running Start ##############################
# Statement of filename
file_train = "data_train.csv" 
file_test = "data_test.csv" 
file_params = "params.json" 
file_out = "result.txt"

# Prepare Data
# Train Data
train_X, train_y = load_data(file_train, 't', 'e')
dtrain = xgb.DMatrix(train_X, label=train_y)
# Test Data
test_X, test_y = load_data(file_test, 't', 'e')
dtest = xgb.DMatrix(test_X, label=test_y)

# Load hyper-parameters
with open(file_params, "r") as f:
    hy_params = json.load(f)

# Get the number of time of interest
K = np.max(np.abs(train_y))

# Hyper Parameters
params = {
    'eta': hy_params['eta'],
    'max_depth': hy_params['max_depth'], 
    'min_child_weight': hy_params['min_child_weight'], 
    'subsample': hy_params['subsample'],
    'colsample_bytree': hy_params['colsample_bytree'],
    'gamma': hy_params['reg_gamma'],
    'lambda': hy_params['reg_lambda'],
    'silent': 1,
    'objective': 'multi:softprob',
    'num_class': K+1,
    'seed': 42
}

# Train model 
# Return empty `watch_list` by default 
# unless list `eval_data` is not empty.
model, watch_list = hitboost(
    dtrain,
    params,
    num_rounds=hy_params['nrounds'],
    L2_alpha=hy_params['L2_alpha'],
    L2_gamma=hy_params['L2_gamma'],
    silent=True
)

# Save the model
# model.save_model("demo.model")

# Feature importance Evaluation
# print model.get_score(importance_type='weight')

# Prediction
pred_train = model.predict(dtrain)
res_train = hit_eval_ci(pred_train, dtrain)
pred_test = model.predict(dtest)
res_test = hit_eval_ci(pred_test, dtest)

# Print Result
print "############# HitBoost ###############"
print "# Evaluation on the training set:"
print "\t%s: %g" % (res_train[0], res_train[1])
print "# Evaluation on the test     set:"
print "\t%s: %g" % (res_test[0], res_test[1])

# Write results into file
if file_out != "":
    with open(file_out, 'w') as f:
        f.write("############## HitBoost ################\n")
        f.write("# Evaluation on the training set:\n")
        f.write("\t%s: %g\n" % (res_train[0], res_train[1]))
        f.write("# Evaluation on the test     set:\n")
        f.write("\t%s: %g\n" % (res_test[0], res_test[1]))