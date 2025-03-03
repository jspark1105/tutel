# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from ..impls.jit_compiler import JitCompiler


def get_kernel_dtype(param_dtype):
  if param_dtype == torch.float16:
      return '__half2'
  elif param_dtype == torch.float32:
      return 'float'
  else:
      raise Exception("Unrecognized data type: %s" % param_dtype)


def create_forward(samples, global_experts, capacity, aligned_dim, param_dtype):
  return JitCompiler.generate_kernel({'capacity': capacity, 'samples': samples, 'hidden': aligned_dim, 'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define capacity (@capacity@)
    #define samples (@samples@)
    #define hidden (@hidden@)
    #define __dtype @dtype@

    extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input) {
      // [thread_extent] blockIdx.x = 128
      // [thread_extent] threadIdx.x = 1024

      for (int i = blockIdx.x; i < samples; i += gridDim.x)
          if (locations1_s[i] < capacity) {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
                  atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], gates1_s[i] * reshaped_input[i * (hidden) + j]);
          }
    }
  ''')


def create_backward_data(samples, global_experts, capacity, aligned_dim, param_dtype):
  return JitCompiler.generate_kernel({'capacity': capacity, 'samples': samples, 'hidden': aligned_dim, 'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define capacity (@capacity@)
    #define samples (@samples@)
    #define hidden (@hidden@)
    #define __dtype @dtype@

    extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s, __dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input) {
      // [thread_extent] blockIdx.x = 128
      // [thread_extent] threadIdx.x = 1024

      for (int i = blockIdx.x; i < samples; i += gridDim.x)
          if (locations1_s[i] < capacity) {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
                  grad_reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
          } else {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
    #if @IS_FLOAT@
                  grad_reshaped_input[i * hidden + j] = __dtype(0);
    #else
                  grad_reshaped_input[i * hidden + j] = __dtype(0, 0);
    #endif
          }
    }
  ''')


def create_backward_gate(samples, global_experts, capacity, aligned_dim, param_dtype):
  return JitCompiler.generate_kernel({'capacity': capacity, 'samples': samples, 'hidden': aligned_dim, 'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define capacity (@capacity@)
    #define samples (@samples@)
    #define hidden (@hidden@)
    #define __dtype @dtype@

    extern "C" __global__ __launch_bounds__(32) void execute(__dtype* __restrict__ dispatched_input, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, void* __restrict__ grad_gates1_s) {
      // [thread_extent] blockIdx.x = @samples@
      // [thread_extent] threadIdx.x = 32
      if (locations1_s[blockIdx.x] >= capacity) {
        if (((int)threadIdx.x) == 0)
    #if @IS_FLOAT@
          ((float*)grad_gates1_s)[(((int)blockIdx.x))] = 0;
    #else
          ((half*)grad_gates1_s)[(((int)blockIdx.x))] = __float2half_rn(0.000000e+00f);
    #endif
        return;
      }
      int indice = indices1_s[(int)blockIdx.x] * capacity + locations1_s[(int)blockIdx.x];
    #if @IS_FLOAT@
      __dtype grad_gates1_s_rf = 0.000000e+00f;
    #else
      __dtype grad_gates1_s_rf = __dtype(0, 0);
    #endif
      for (int i = threadIdx.x; i < hidden; i += 32)
        grad_gates1_s_rf += dispatched_input[indice * (hidden) + i] * reshaped_input[((int)blockIdx.x) * (hidden) + i];

    #if !defined(__HIPCC__)
      __dtype red_buf0[1];
      unsigned int mask[1];
      __dtype t0[1];
      red_buf0[(0)] = grad_gates1_s_rf;
      mask[(0)] = __activemask();
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]);
      red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
    #else
      __shared__ __dtype red_buf0[32];
      __syncthreads();
      ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = grad_gates1_s_rf;
      if (((int)threadIdx.x) < 16) {
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 16))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 8))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 4))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 2))]));
        ((volatile __dtype*)red_buf0)[(((int)threadIdx.x))] = ((__dtype)(((volatile __dtype*)red_buf0)[(((int)threadIdx.x))]) + (__dtype)(((volatile __dtype*)red_buf0)[((((int)threadIdx.x) + 1))]));
      }
      __syncthreads();
    #endif
      if (((int)threadIdx.x) == 0)
    #if @IS_FLOAT@
        ((float*)grad_gates1_s)[(((int)blockIdx.x))] = red_buf0[(0)];
    #else
        ((half*)grad_gates1_s)[(((int)blockIdx.x))] = red_buf0[(0)].x + red_buf0[(0)].y;
    #endif
    }
  ''')
