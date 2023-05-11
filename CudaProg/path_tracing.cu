//#include "DeviceKernel.cuh"
#include <cmath>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cooperative_groups.h>

