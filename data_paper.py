# Numeric example in the paper
# Parameters
J = 30  # Number of tasks
M = 5   # Number of mated stations
N = 3   # Number of product models
A = 20  # Number of agents
H = [1, 2, 3, 4]  # Sides
robots = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  
humans = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}  
ls = [1, 2, 3, 4]
bs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
rs = [16, 17, 18, 19, 20]
us = [21, 22, 23, 24, 25, 26]
ps = [27, 28, 29, 30]
tb = [1, 3, 4, 6, 8, 10, 12, 14, 15, 17, 19]
to = [2, 5, 7, 13, 18]
tr = [9, 11, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
MU = [2, 3]
MP = [3, 4]
f_mh = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]
cy = 100
phi = 999000000
mu_1 = 0.6
mu_2 = 0.3
C = {a: random.uniform(30, 50) for a in range(A)}
t_ajn = {(j, n, a): random.uniform(3, 15) for j in range(J) for a in range(A) for n in range(N)}

# Relationships between parameters
po = {(1, 2)} # Pairs for positive zoning constraints
ne = {(1, 30), (1, 5)}  # Pairs for negative zoning constraints
se = {(4, 6)}  # Pairs for synchronous constraints
sa = {(2, 8), (2, 9), (4, 11), (5, 10), (5, 30), (9, 12), (9, 14), (11, 12), (11, 16), 
      (15, 23), (15, 30), (2, 12), (2, 14), (4, 12), (4, 16)}  # Successor relationships
pa = {(8, 2), (9, 2), (11, 4), (10, 5), (30, 5), (12, 9), (14, 9), (12, 11), (16, 11),
      (23, 15), (30, 15), (12, 2), (14, 2), (12, 4), (16, 4)}  # Predecessor relationships

  