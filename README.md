# HitBoost
Survival Analysis via a Multi-output Gradient Boosting Decision Tree method.

More details can refer to [this](https://doi.org/10.1109/ACCESS.2019.2913428) that has been published in IEEE Access. Citation could be found in [#Citation](#citation).

If you have any questions, feel free to submit your issues.

### Requirements

Requirements:

- Python version: `2.x`. (`3.x` will be supported later)
- Packages Dependency: `xgboost`, `numpy`, `pandas`, `json`.


### Introduction

Function for building `HitBoost` model:

```Python
hitboost(
    dtrain, 
    model_params, 
    num_rounds=100, 
    L2_alpha=1.0, 
    L2_gamma=0.01, 
    eval_data=[], 
    silent=True, 
    save_model_as=""
)
```

Arguments Explanation:

- `dtrain`: `xgb.DMatrix` object. Training data.
- `model_params`: `dict`. Parameters of xgboost multi-classification model.
- `num_rounds`: `int`. The number of iterations.
- `L2_alpha`: `float`. The coefficient of L2 term in objective function.
- `L2_gamma`: `float`. The parameter in L2 term.
- `eval_data`: `list`. Data (`xgb.DMatrix`) used to be evaluated.
- `silent`: `bool`. Whether print info on screen.
- `save_model_as`: `str`. Path for saving fitted model.

Complete `model_params`:

```
model_params = {
    'eta': 0.1,
    'max_depth':3, 
    'min_child_weight': 8, 
    'subsample': 0.9,
    'colsample_bytree': 0.5,
    'gamma': 0,
    'lambda': 0,
    'silent': 1,
    'objective': 'multi:softprob',
    'num_class': K+1,
    'seed': 42
}
```

Runing `HitBoost`: 

- See more in `core/demo.py`.

### TODO

In the future, some features that need to be advanced in this repo are as follows:
- support for `python 3.x`
- more stability of the customized loss function
- docs of usage

### Citation

- P. Liu, B. Fu and S. X. Yang, "HitBoost: Survival Analysis via A Multi-output Gradient Boosting Decision Tree Method," in IEEE Access. doi: 10.1109/ACCESS.2019.2913428