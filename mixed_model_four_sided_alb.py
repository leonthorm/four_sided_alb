from model import four_sided_alb_model
from gurobipy import *

if __name__ == '__main__':

    model = four_sided_alb_model
    model.optimize()