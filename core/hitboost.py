# coding=utf-8
import xgboost as xgb
from hitcore import *

def hit_eval(model, eval_data=[]):
    loss_list = []
    ci_list = []
    for d in eval_data:
        pred_d = model.predict(d)
        lossv = hit_eval_loss(pred_d, d)[1]
        civ = hit_eval_ci(pred_d, d)[1]
        loss_list.append(lossv)
        ci_list.append(civ)
    return loss_list, ci_list

def print_eval(iters_num, loss_list, ci_list):
    print "# After %dth iteration:" % iters_num
    for i in range(len(loss_list)):
        print "\tOn %d-th data:" % (i + 1)
        print "\t\tLoss: %g" % loss_list[i]
        print "\t\ttd-CI: %g" % ci_list[i]

def hitboost(dtrain, model_params, num_rounds=100, 
             L2_alpha=1.0, L2_gamma=0.01, 
             eval_data=[], silent=True, save_model_as=""):
    """
    Fitting hit boosting model.

    params
    ------
    dtrain: `xgb.DMatrix` object
        Training data.
    model_params: `dict`
        Parameters of xgboost multi-classification model.
        For example:
            params = {
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
    num_rounds: `int`
        The number of iterations.
    L2_alpha: `float`
        The coefficient of L2 term in objective function.
    L2_gamma: `float`
        The parameter in L2 term.
    eval_data: `list`
        Data (xgb.DMatrix) used to be evaluated.
    silent: `bool`
        Print infos to screen.
    save_model_as: `str`
        Path for saving fitted model.
    """
    eval_result = {'td-CI': [], 'Loss': []}
    global_init(L2_gamma, L2_alpha)
    model = xgb.Booster(model_params, [dtrain])
    for _ in range(num_rounds):
        # Note: Since default setting of `output_margin` is `False`,
        # so the prediction is outputted after softmax transformation.
        pred = model.predict(dtrain)
        # Note: The gradient you provide for `model.boost()` must be 
        # gradients of objective function with respect to the direct 
        # output of boosting tree (even if you set `output_margin` as 
        # `True`).
        g, h = hit_grads(pred, dtrain)
        model.boost(dtrain, g, h)
        # Append to eval_result
        if len(eval_data) > 0:
            res_loss, res_ci = hit_eval(model, eval_data)
            eval_result['Loss'].append(res_loss)
            eval_result['td-CI'].append(res_ci)
            if not silent:
                print_eval(_ + 1, res_loss, res_ci)
    # Model saving
    if save_model_as != "":
        model.save_model(save_model_as)
    # Return model and evaluation results
    return model, eval_result