import gurobipy as gp
from gurobipy import GRB
from gurobipy import *  # imports everything from gurobipy without alias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def four_sided_alb_model() -> gp.Model:
    model = Model("MILP")


    # sets
    tasks = []  # Set of tasks
    mated_stations = []  # Set of mated stations
    sides = ['left-side', 'right-side', 'beneath-side', 'above-side']
    agents = []  # Set of agents
    agents_human = []  # Index of humans in the subset of agents
    agents_robot = []  # Index of robots in the subset of agents
    product_models = []  # Set of product models

    # Specific Task Sets
    ls = []  # Tasks performed on the left side
    rs = []  # Tasks performed on the right side
    bs = []  # Tasks performed on either left or right
    us = []  # Tasks performed beneath-side
    ps = []  # Tasks performed above-side
    to = []  # Tasks only humans can do
    tr = []  # Tasks only robots can do
    tb = []  # Tasks both humans and robots can do

    t = {} #  processing time of task j for model n by agent a

    # Precedence and Zoning
    pi = {}  # Immediate predecessors
    sa = {}  # Immediate successors
    po = []  # Positive zoning
    ne = []  # Negative zoning
    se = []  # Synchronous tasks

    C = []

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
    tf = model.addVars(J, N, vtype=GRB.CONTINUOUS, name = 'Finishing time of task j for model n')
    rr = model.addVars(J, N, vtype=GRB.CONTINUOUS, name = 'Real time of task j for model n' )
    alpha = model.addVars(M, vtype=GRB.BINARY, name = 'just one side of mated-station is utilized')
    beta = model.addVars(M, vtype=GRB.BINARY, name = 'two sides of mated-station are utilized')
    theta = model.addVars(M, vtype=GRB.BINARY, name = 'three sides of mated-station are utilized')
    gamma = model.addVars(M, vtype=GRB.BINARY, name = 'four sides of mated-station are utilized')
    v1 = model.addVars(M,  H, vtype=GRB.BINARY, name = 'at least one task to in mated-station m with side h')
    v2 = model.addVars(M,  H, vtype=GRB.BINARY, name = 'at least one task tr in mated-station m with side h')
    cy = model.addVar(lb=0, name="cycle time non negative")

    mu = 10e8
    mu1 = 1 # the opened mated-station weight
    mu2 = 1 # The opened station weight
    # 1
    model.setObjective(
        mu1 * quicksum((alpha[m] + beta[m] + theta[m] + gamma[m]) for m in range(M)) + mu2 * quicksum(quicksum(z[m,h] for h in range(H)) for m in range(M)),
        GRB.MINIMIZE
    )
    # 2
    model.setObjective(
        quicksum((C[a]*(quicksum(quicksum(q[a,h,m] for m in range(M)) for h in range(H)))) for a in range(A)),
        GRB.MINIMIZE
    )
    # 3
    model.addConstrs((quicksum(quicksum(x[j, m, h] for h in range(H)) for m in range(M)) == 1 for j in range(J)), name="Task_Assignment")

    # Ensure tasks are distributed across stations more evenly - chatgpt
    model.addConstrs(
    gp.quicksum(x[j, m, h] for h in range(H)) <= (J / M) + 1  # Soft balancing constraint
    for j in range(J) for m in range(M)
    )

    
    # 4
    # TODO subsets
    model.addConstrs((quicksum(x[j, m, 2] for m in range(M)) == 1 for j in us), name="Beneath_Task_Assignment")
    # 5
    # TODO subsets
    model.addConstrs((quicksum(x[j, m, 3] for m in range(M)) == 1 for j in ps), name="Above_Task_Assignment")
    # 6
    model.addConstrs((quicksum(quicksum(q[a, h, m] for h in range(H)) for m in range(M)) <= 1 for a in range(A)), name="Agent_Assignment")

    # Ensure only humans do tasks in 'to' (tasks only humans can do) - chatgpt
    model.addConstrs(
    gp.quicksum(q[a, h, m] for a in humans) >= gp.quicksum(x[j, m, h] for j in to for h in range(H)) 
    for m in range(M)
    )

    # Ensure only robots do tasks in 'tr' (tasks only robots can do) - chatgpt
    model.addConstrs(
    gp.quicksum(q[a, h, m] for a in robots) >= gp.quicksum(x[j, m, h] for j in tr for h in range(H)) 
    for m in range(M)
    )
    # 7
    model.addConstrs(quicksum(q[a, h, m] for a in range(A)) == z[m, h] for h in range(H) for m in range(M))
    # 8
    model.addConstrs(quicksum(q[a, 4-1, m] for a in range(A)) <= 0 for m in agents_human)
    # 9
    model.addConstrs(tf[j, n] <= cy for j in range(J) for n in range(N))
    # 10
    model.addConstrs(tf[j, n] >= rr[j, n] for j in range(J) for n in range(N))
    
    # Ensure tasks finish within cycle time - chatgpt
    model.addConstrs(
    tf[j, n] <= cy for j in range(J) for n in range(N)
    )

    # Update the objective function to minimize cycle time - chatgpt
    model.setObjective(
    cy + gp.quicksum(C[a] * q[a, h, m] for a in range(A) for h in range(H) for m in range(M)), 
    GRB.MINIMIZE
    )

    # 11
    model.addConstrs(
        (rr[j, n]
         == quicksum(quicksum(quicksum(t[a, j, n] * x[j, m, h] * q[a, h, m] for m in range(M)) for a in range(A)) for h in range(H)))
        for j in range(J) for n in range(N)
    )
    # 12
    # TODO subsets
    model.addConstrs(
        quicksum(quicksum(s * x[i, s, h] for h in range(H)) for s in range(M))
         <= quicksum(quicksum(m * x[j, m, h] for h in range(H)) for m in range(M))
        for j in range(J) for i in range(pi)
    )
    # 13
    # TODO subsets
    model.addConstrs(
        tf[j, n] - tf[i, n] + mu*(1-quicksum([i, m, h] for h in range(H))) + mu*(1-quicksum([j, m, h] for h in range(H)))
        >= rr[j, n]*N for j in range(J) for i in range(pi) for m in range(M) for n in range(N)
    )
    # 14
    # TODO subsets
    model.addConstrs(
        (tf[k, n] - tf[j, n] + mu * (1 - x[j, m, h]) + (1 - x[k, m, h]) + mu * (1 - y[j, k])
         >= rr[k, n] * N)
        for j in range(J) for n in range(N) for k in range(K) for m in range(M) for h in range(H)
    )
    # 15
    model.addConstrs(
        (tf[j, n] - tf[k, n] + mu * (1 - x[j, m, h]) + (1 - x[k, m, h]) + mu * y[j, k]
         >= rr[j, n])
        for j in range(J) for n in range(N) for k in range(K) for m in range(M) for h in range(H)
    )
    # 16
    model.addConstrs(
        quicksum(x[j, m, h] for j in range(J)) - mu * z[m, h] <= 0 for m in range(M) for h in range(H)
    )
    # 17
    model.addConstrs(
        quicksum(z[m, h] for h in range(H)) - 4 * gamma[m] - 3 * theta[m] - 2 * beta[m] - alpha[m] == 0 for m in range(M)
    )
    # 18
    # todo check for m
    model.addConstrs(
        x[j,m,h] - x[i,m,h] == 0 for (j,i) in po for h in range(H) for m in range(M)
    )
    # 19
    model.addConstrs(
        quicksum(x[j,m,h] for h in range(H)) + quicksum(x[i,m,h] for h in range(H)) <= 0 for (j,i) in ne for m in range(M)
    )
    # 20
    # todo add h != l
    model.addConstrs(
        x[j,m,l] - x[i,m,h] == 0 for (j,i) in se for h in range(H) for l in range(H) for m in range(M)
    )
    # 21
    model.addConstrs(
        tf[j, n] - rr[j, n] == tf[i,n] - rr[i,n] for (j,i) in se for n in range(N)
    )
    # 22
    model.addConstrs(
        quicksum(x[j,m,h] for j in to) <= mu * v1[m,h] for m in range(M) for h in range(H)
    )
    model.addConstrs(
        quicksum(x[j,m,h] for j in to) >= v1[m,h] for m in range(M) for h in range(H)
    )
    # 23
    model.addConstrs(
        quicksum(x[j, m, h] for j in tr) <= mu * v2[m, h] for m in range(M) for h in range(H)
    )
    model.addConstrs(
        quicksum(x[j, m, h] for j in tr) >= v2[m, h] for m in range(M) for h in range(H)
    )
    # 24
    # todo subset
    model.addConstrs(
        quicksum(q[a,h,m] for a in range(A)) <= 1 - (v1[m,h]) for m in range(M) for h in range(H)
    )
    # 25
    # todo subset
    model.addConstrs(
        quicksum(q[a, h, m] for a in range(A)) <= 1 - (v2[m, h]) for m in range(M) for h in range(H)
    )

    # Enforce strict binary variable constraints - chatgpt
    model.Params.IntFeasTol = 1e-6



    return model
