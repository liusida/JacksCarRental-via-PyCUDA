/**
 * Copyright (C)
 * 2020 Sida Liu(learner.sida.liu@gmail.com)
 * Permission given to modify the code as long as you keep this declaration at the top
 */

/**
 * This file is not a standalone CUDA source file. It should be used together with `car_rental_cuda.py`.
 * The pre-defined constant variables are there.
 */

#include "cuda.h"

/**
 * Calculate Expected Return (in dollar)
 *
 * @param[in] state_1 The number of cars in the first location
 * @param[in] state_2 The number of cars in the second location
 * @param[in] action The number of cars moved from the first location to the second location
 * @param[in/out] state_value The Value Matrix
 * @param[in] num_state The number of states
 * 
 * There is still a loop (POISSON_UPPER_BOUND^4) in this function. It is summing over all the state. It is slower to do Dynamic Parallelism here.
 */
__device__ float expected_return(unsigned int state_1, unsigned int state_2, int action, float *state_value, unsigned int num_state) {
  const int idx = state_1 * num_state + state_2;
  float new_value = 0.0;

  new_value -= MOVE_CAR_COST * abs(action);

  const int NUM_OF_CARS_FIRST_LOC = min(state_1 - action, MAX_CARS);
  const int NUM_OF_CARS_SECOND_LOC = min(state_2 + action, MAX_CARS);

  float total_prob = 0.0;
  for (int rental_request_first_loc = 0; rental_request_first_loc < POISSON_UPPER_BOUND; rental_request_first_loc++) {
    for (int rental_request_second_loc = 0; rental_request_second_loc < POISSON_UPPER_BOUND; rental_request_second_loc++) {
      float prob = poisson_pmf_req_1[rental_request_first_loc] * poisson_pmf_req_2[rental_request_second_loc];

      int num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC;
      int num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC;

      int valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc);
      int valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc);

      float reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT;

      num_of_cars_first_loc -= valid_rental_first_loc;
      num_of_cars_second_loc -= valid_rental_second_loc;

      for (int returned_cars_first_loc = 0; returned_cars_first_loc < POISSON_UPPER_BOUND; returned_cars_first_loc++) {
        for (int returned_cars_second_loc = 0; returned_cars_second_loc < POISSON_UPPER_BOUND; returned_cars_second_loc++) {
          float prob_return = poisson_pmf_ret_1[returned_cars_first_loc] * poisson_pmf_ret_2[returned_cars_second_loc];
          int num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS);
          int num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS);
          int idx_ = num_of_cars_first_loc_ * num_state + num_of_cars_second_loc_;
          float prob_ = prob * prob_return;
          total_prob += prob_;
          new_value += prob_ * (reward + DISCOUNT * (state_value[idx_]));
        }
      }
    }
  }
  // Sanity Check: make sure total probability is close to 1.0
  // printf(" total_prob: %f %f \n", total_prob, poisson_pmf_req_1[0]);
  return new_value;
}

/**
 * Evaluate the Policy, update the Value Matrix following the current Policy Matrix
 *
 * @param[in/out] state_value The Value Matrix
 * @param[in] state_policy The Policy Matrix
 * @param[in] num_state The number of states
 */
__global__ void gpu_policy_evaluation(float *state_value, int *state_policy, unsigned int num_state) {
  int a = threadIdx.x, b = threadIdx.y;
  int idx = a * num_state + b;
  const int action = state_policy[idx];

  state_value[idx] = expected_return(a, b, action, state_value, num_state);
}

/**
 * Improve the Policy, update the Policy Matrix according the current Value Matrix
 *
 * @param[in] state_value The Value Matrix
 * @param[in/out] state_policy The Policy Matrix
 * @param[in] num_state The number of states
 */
__global__ void gpu_policy_improvement(float *state_value, int *state_policy, unsigned int num_state) {
  int a = threadIdx.x, b = threadIdx.y;
  int idx = a * num_state + b;
  int old_action = state_policy[idx];
  int new_action = -MAX_MOVE_OF_CARS - 1;
  float max_value = 0.0, value = 0.0;
  for (int action = -MAX_MOVE_OF_CARS; action < MAX_MOVE_OF_CARS + 1; action++) {
    if ((0 <= action && action <= a) || (-b <= action && action <= 0)) {
      value = expected_return(a, b, action, state_value, num_state);
    } else {
      value = -999999;
    }
    if (max_value < value) {
      max_value = value;
      new_action = action;
    }
  }
  state_policy[idx] = new_action;
}