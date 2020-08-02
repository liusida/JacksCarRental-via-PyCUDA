#######################################################################
# Copyright (C)                                                       #
# 2020 Sida Liu(learner.sida.liu@gmail.com)                           #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#
# Example 4.2 in book Reinforcement Learning (2nd Edition), Sutton, Page 81
# This implementation is a translation of the original `car_rental.py` to CUDA
# The original repo: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction
# The book: http://www.incompleteideas.net/book/the-book-2nd.html
#
# The Problem:
#   Jack has two locations of a car company.
#   The number of car requested: num_req ~ poisson(lam), lam_req_1=3, lam_req_2=4
#   The number of car returned: num_ret ~ poisson(lam), lam_ret_1=3, lam_ret_2=2
#   Let state be the tuple of numbers of cars at two locations, state_i = <num_1, num_2>
#   Let action be the number of cars moved from location 1 to location 2, action_i = num_moved
#   Let reward be the money, reward_i = $10 x ( min(num_req_1, num_1) + min(num_req_2, num_2) ) - $2 x num_moved
#   Init value table and policy table to be all 0
#
# CUDA Requirements:
#   Test with CUDA 10.1 and pycuda 2019.1.2
#
# Results:
#   Results are in the images subfolder.
#   Note that the result of the original version was calculated using constant returned cars instead of possion variable
#   Execution time (with Nvidia GeForce GTX 1070 & Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz):
#       The original version: ~ 1 hour
#       The original version with const returned cars: 1 min
#       The CUDA version: 6 sec
#
# We can see in this example, CUDA brought at least 600x speedup.
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import DynamicSourceModule, SourceModule
# Notice: all pycuda variables should be explicitly 32-bit

matplotlib.use('Agg')

# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a car
MOVE_CAR_COST = 2

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# all possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)


def float_array_to_str(arr):
    """
    Turn a float array into CUDA code string
    @arr: array of float numbers (e.g. [1.0, 2.0])
    return: a string (e.g. "{1.0,2.0}")
    """
    s = "{"
    for f in arr:
        s += str(f) + ","
    s = s[:-1] + "}"
    return s


def poisson_cache_string(lam=3, upbound=POISSON_UPPER_BOUND):
    """
    Building a string of Poisson PMF array for CUDA code
    return: The PMF of Poisson Distribution with lambda
    """
    arr = [poisson.pmf(n, lam) for n in range(upbound)]
    np_arr = np.array(arr, dtype=np.float32)
    return float_array_to_str(np_arr)

# The Poisson PMF with different lambda used in this example
poisson_pmf_2_str = poisson_cache_string(2)
poisson_pmf_3_str = poisson_cache_string(3)
poisson_pmf_4_str = poisson_cache_string(4)

# Value matrix
num_state = MAX_CARS + 1
num_state_gpu = np.uint32(num_state)
state_value = np.zeros((num_state, num_state), dtype=np.float32)
state_value_gpu = gpuarray.to_gpu(state_value)
# Policy matrix
state_policy = np.zeros(state_value.shape, dtype=np.int32)
state_policy_gpu = gpuarray.to_gpu(state_policy)

# Prepare the CUDA source code for pre-defined constant variables
cuda_source = f"""
        const float RENTAL_CREDIT = {RENTAL_CREDIT};
        const float MOVE_CAR_COST = {MOVE_CAR_COST};
        const float DISCOUNT = {DISCOUNT};
        const unsigned int MAX_CARS = {MAX_CARS};
        const int MAX_MOVE_OF_CARS = {MAX_MOVE_OF_CARS};
        const unsigned int POISSON_UPPER_BOUND = {POISSON_UPPER_BOUND};
        __constant__ float poisson_pmf_req_1[] = {poisson_pmf_3_str};
        __constant__ float poisson_pmf_req_2[] = {poisson_pmf_4_str};
        __constant__ float poisson_pmf_ret_1[] = {poisson_pmf_3_str};
        __constant__ float poisson_pmf_ret_2[] = {poisson_pmf_2_str};
    """
# Additional to the constant variables, attach the source file
with open(f"{__file__}.cu", "r") as f:
    cuda_source += f.read()

# Compile CUDA source code and expose functions to Python
cuda_kernel = SourceModule(cuda_source)
# If we want to use Dynamic Parallelism, we should use DynamicSourceModule:
#   cuda_kernel = DynamicSourceModule(cuda_source, cuda_libdir='/usr/local/cuda/lib64')
gpu_policy_evaluation = cuda_kernel.get_function("gpu_policy_evaluation")
gpu_policy_improvement = cuda_kernel.get_function("gpu_policy_improvement")


def policy_evaluation():
    """
    Estimate state value under current state policy.
    Keep updating state value until converged. (sequentially)
    """
    global state_value, state_value_gpu
    while True:
        old_value = state_value.copy()

        # Start parallelize here.
        # Notice: always align the parameters. Mistakes in order will cause error message such as "illegal memory access"
        gpu_policy_evaluation(
            state_value_gpu, state_policy_gpu,
            num_state_gpu,
            block=(num_state, num_state, 1), grid=(1, 1), shared=0)

        state_value = state_value_gpu.get()
        max_value_change = abs(old_value - state_value).max()
        print('max value change {}'.format(max_value_change))
        if max_value_change < 1e-4:
            break


def policy_improvement():
    """
    Update state policy according to current state value.
    return: whether policy is stablized.
    """
    global state_policy, state_policy_gpu
    old_policy = state_policy.copy()
    gpu_policy_improvement(
        state_value_gpu, state_policy_gpu,
        num_state_gpu,
        block=(num_state, num_state, 1), grid=(1, 1), shared=0)
    state_policy = state_policy_gpu.get()
    max_policy_change = abs(old_policy - state_policy).max()
    policy_stable = max_policy_change < 1
    print(f"max_policy_change: {max_policy_change}")
    return policy_stable


def figure_4_2():
    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(state_policy),
                          cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)
        policy_evaluation()

        # policy improvement
        if policy_improvement():
            print('policy stable: True')
            fig = sns.heatmap(np.flipud(state_value),
                              cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break
        iterations += 1

    plt.savefig('images/figure_4_2_cuda.png')
    plt.close()


if __name__ == '__main__':
    figure_4_2()
