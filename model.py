import gurobipy as gp
from gurobipy import GRB
from gurobipy import *  # imports everything from gurobipy without alias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def four_sided_alb_model() -> gp.Model:
    model = Model("MILP")


    # indices
    tasks = []
    mated_stations = []
    sides = ['left−side', 'right−side', 'beneath−side', 'above−side']
    agents = []
    product_models = []

    J, I, K = len(tasks)
    M, S = len(mated_stations)
    N = len(product_models)
    H, L = len(sides)
    A = len(agents)

    # Variables
    x = model.addVars(J, M, H, vtype=GRB.BINARY, name = 'task j is assigned to mated-station m with side h')
    z = model.addVars(M, H, vtype=GRB.BINARY, name = 'station m is utilized for a model')
    y = model.addVars(J, K, vtype=GRB.BINARY, name = 'task j is assigned earlier than task k in the same station')
    q = model.addVars(A, H, M, vtype=GRB.BINARY, name = 'agent a is assigned to side h in mated-station m')
    t_finish = model.addVars(J, N, vtype=GRB.CONTINUOUS, name = 'Finishing time of task j for model n')
    rr = model.addVars(J, N, vtype=GRB.CONTINUOUS, name = 'Real time of task j for model n' )
    alpha = model.addVars(M, vtype=GRB.BINARY, name = 'just one side of mated-station is utilized')
    beta = model.addVars(M, vtype=GRB.BINARY, name = 'two sides of mated-station are utilized')
    theta = model.addVars(M, vtype=GRB.BINARY, name = 'three sides of mated-station are utilized')
    gamma = model.addVars(M, vtype=GRB.BINARY, name = 'four sides of mated-station are utilized')
    v1 = model.addVars(M,  H, vtype=GRB.BINARY, name = 'at least one task t_o in mated-station m with side h')
    v2 = model.addVars(M,  H, vtype=GRB.BINARY, name = 'at least one task t_r in mated-station m with side h')


    return model