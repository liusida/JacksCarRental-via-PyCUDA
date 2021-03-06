# How to use PyCUDA to bring significant speedup

Imagine that we have designed an computational experiment in Python, and we waited 3 days for the results, and after that, unfortunately we discovered there was a typo or a small bug in the source code. What do you think we would say when we restart the experiment? I would hope that the experiment could be run in half a hour.

It is possible, by making the code parallized.

CUDA is a C++-like program language for parallel programs which can run on Nvidia GPU. (https://developer.nvidia.com/cuda-toolkit)

PyCUDA is an open source Python interface to compile CUDA source code on the fly and execute it. (https://documen.tician.de/pycuda/)

Here we show an example of using CUDA and PyCUDA to rewrite a Python program.

The file `car_rental.py` is a Python program. It is slow because there are huge nested loops. We can exam this by searching for the keywords `while True` and `for`.

The file `car_rental_cuda.py` is the CUDA-optimized version of the original program. The `gpu_policy_evaluation` and `gpu_policy_improvement` are two kernels (CUDA interfaces) that can run 21*21 (num_state=21) threads in parallel. In this code, it prepares the pre-defined constant vairables and read in the CUDA source file `car_rental_cuda.py.cu`, compiles them on the fly, and expose the interfaces as Python functions.

By running them, we can get the results in the `images/` folder. And we can see the CUDA version only takes 6 seconds while the original version would take more than a hour.

