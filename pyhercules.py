import hercules
import random
import time

# set a random seed
random.seed(time.time())

# create a random initial point
x_0 = [random.randint(0, 1) for i in range(1000)]

# solve the QUBO problem, by reading from a file and solving it from the initial point x_0, for 30 iterations
x_soln, obj = hercules.mls_from_file('test_large.qubo', x_0, 30)

# print the objective value of the solution
print('Solution: ', obj)
