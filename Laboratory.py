#from metaheuristiscs_master.models.multiple_solution.swarm_based.WOA import BaoWOA
from IWOA import BaseWOA, BaoWOA
from utils.FunctionUtil import whale_f1 #, whale_f2, whale_f3, whale_f4, whale_f5, whale_f6, whale_f7, whale_f8, whale_f9, whale_f10, whale_f11, square_function

## Setting parameters`
root_paras = {
    "problem_size": 12,
    "domain_range": [-50, 50],
    "print_train": True,
    "objective_func": whale_f1
}
woa_paras = {
    "epoch": 500,
    "pop_size": 10
}

## Run model
md = BaoWOA(root_algo_paras=root_paras, woa_paras=woa_paras)
md._train__()


