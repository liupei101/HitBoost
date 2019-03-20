# HitBoost
Survival Analysis via a Multi-output Gradient Boosting Decision Tree method

### Running Requirements

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
