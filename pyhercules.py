import hercules
import random
import time

# set a random seed
random.seed(time.time())

# read in the qubo problem
problem = hercules.read_qubo('test_large.qubo')
num_x = problem[-1]

# create a random initial point
x_0 = [random.randint(0, 1) for i in range(num_x)]

# start timing
start = time.time()

# solve the QUBO problem via the mixed local search heuristic with, initial point x_0, for 50 iterations
x_soln, obj = hercules.mls(problem, x_0, 50)

# stop timing
end = time.time()

# print the objective value of the solution
print('Solution: ', obj, ' in ', end - start, ' seconds')
