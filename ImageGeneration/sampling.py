# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import einops
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

import matplotlib.pyplot as plt

import torchvision
from tqdm import tqdm


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'rectified_flow':
    sampling_fn = get_rectified_flow_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda'):
  """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def euler_sampler(model, z=None):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False) 
      
      ### Uniform
      dt = 1./sde.sample_N
      eps = 1e-3 # default: 1e-3
      for i in range(sde.sample_N):
        
        num_t = i /sde.sample_N * (sde.T - eps) + eps
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999) ### Copy from models/utils.py 

        # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
        sigma_t = sde.sigma_t(num_t)
        pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

        x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
      
      x = inverse_scaler(x)
      nfe = sde.sample_N
      return x, nfe
  
  def rk45_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False)

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)

        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      x = inverse_scaler(x)
      
      return x, nfe
  
  def parallel_sampler(model, z=None):
    block_size = 2
    
    for block_size in [2, 4, 6, 8]:
      with torch.no_grad():
        if z is None:
          z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
          x = z0.detach().clone()
        else:
          x = z
        
        # return x, 1
        nfe_ctr = 0
        
        model_fn = mutils.get_model_fn(model, train=False)
        
        buffer = torch.stack([x] * (sde.sample_N + 1))
        # inv_sigmas = 1. / torch.stack([sde.sigma_t(num_t) for num_t in range(0, sde.sample_N)])
        # inv_sigmas = inv_sigmas / x[0].numel

        # print(buffer.shape)
        begin_idx = 0
        end_idx = block_size
        eps = 1e-3 # default: 1e-3
        dt = 1./sde.sample_N

        while begin_idx < sde.sample_N:
          block_latents = buffer[begin_idx:end_idx]
          # reshape block_latents from block, batch, c, h, w to block * batch, c, h , w
          block_latents = einops.rearrange(block_latents, 'bl ba c h w -> (bl ba) c h w').to(device)
          # print(block_latents.shape)

          t = (torch.stack([torch.ones(shape[0], device=device) * num_t for num_t in range(begin_idx, end_idx)]) / sde.sample_N * (sde.T - eps) + eps) * 999
          t = einops.rearrange(t, 'bl ba -> (bl ba)').to(device)
          # print(t.shape)
          # print(t)
          # print('sigma', sde.sigma_t(t / 999))
          # inv_sigmas = 1. / sde.sigma_t(t / 999)
          # inv_sigmas = inv_sigmas / x[0].numel()
          # print('inv_sigmas', inv_sigmas)

          pred = model_fn(block_latents, t)
          nfe_ctr += end_idx - begin_idx

          # print('pred', pred.shape)
          # diff = pred - block_latents
          diff = pred * dt
          diff_ex = einops.rearrange(diff, '(bl ba) c h w -> bl ba c h w', bl=end_idx-begin_idx)
          # print('diff_ex', diff_ex.shape)
          cum_diff = torch.cumsum(diff_ex, dim=0)
          
          block_latents_new = buffer[begin_idx][None,] + cum_diff
          # print('block_latents_new shape', block_latents_new.shape)
          cum_error_vec = (block_latents_new - buffer[begin_idx + 1: end_idx +1])
          cur_error = (torch.linalg.norm(cum_error_vec.reshape(cum_error_vec.shape[0], cum_error_vec.shape[1], -1), dim=-1)).pow(2)
          # print('cur error', cur_error.shape)
          error_ratio = cur_error #* inv_sigmas.reshape(cur_error.size())
          # print('err_ratio', error_ratio)
          error_ratio = torch.nn.functional.pad(
              error_ratio, (0, 0, 0, 1), value=1e9
          )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
          any_error_at_time = torch.max(error_ratio > 0.1 ** 2, dim=1).values.int()
          ind = torch.argmax(any_error_at_time).item()
          # print(ind)

          new_begin_idx = begin_idx + min(1 + ind, block_size)
          new_end_idx = min(new_begin_idx + block_size, sde.sample_N)
          
          buffer[begin_idx + 1: end_idx + 1] = block_latents_new
          # linear extrap fill
          def get_deriv(b, e):
            deriv = (buffer[e] - buffer[b]) / (e - b)
            return deriv
        
          # deriv = (get_deriv(end_idx - 1, end_idx) + get_deriv(end_idx - 2, end_idx)) / 2
          deriv = (get_deriv(end_idx - 1, end_idx))
          buffer[end_idx: new_end_idx + 1] = buffer[end_idx][None,]
          if new_end_idx > end_idx:
            buffer[end_idx: new_end_idx + 1] += deriv * torch.arange(new_end_idx + 1 - end_idx)[:, None, None, None, None].to(device)
          
          begin_idx = new_begin_idx
          end_idx = new_end_idx
        print("NFE", nfe_ctr, 'block', block_size)

    x = inverse_scaler(buffer[-1])
    return x, nfe_ctr
        
  def strided_parallel_sampler(model, z=None):
    bl_size=2
    stride=2
    block_size=stride * bl_size
    with torch.no_grad():
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      # return x, 1
      nfe_ctr = 0
      
      model_fn = mutils.get_model_fn(model, train=False)
      
      buffer = torch.stack([x] * (sde.sample_N + 1))

      # print(buffer.shape)
      begin_idx = 0
      end_idx = block_size
      eps = 1e-3 # default: 1e-3
      dt = 1./sde.sample_N

      while begin_idx < sde.sample_N:
        if end_idx == sde.sample_N:
          stride = 1
          end_idx = min(begin_idx + bl_size, sde.sample_N)
        block_latents = buffer[begin_idx:end_idx:stride]
        # reshape block_latents from block, batch, c, h, w to block * batch, c, h , w
        block_latents = einops.rearrange(block_latents, 'bl ba c h w -> (bl ba) c h w').to(device)
        # print(block_latents.shape)

        t = (torch.stack([torch.ones(shape[0], device=device) * num_t for num_t in range(begin_idx, end_idx, stride)]) / sde.sample_N * (sde.T - eps) + eps) * 999
        t = einops.rearrange(t, 'bl ba -> (bl ba)').to(device)

        pred = model_fn(block_latents, t)
        nfe_ctr += (end_idx - begin_idx) // stride

        # print('pred', pred.shape)
        # diff = pred - block_latents
        diff = pred * dt
        diff_ex = einops.rearrange(diff, '(bl ba) c h w -> bl ba c h w', bl=(end_idx-begin_idx) // stride)
        diff_ex = einops.repeat(diff_ex, 'bl ba c h w -> (bl s) ba c h w', s=stride)

        # print('diff_ex', diff_ex.shape)
        cum_diff = torch.cumsum(diff_ex, dim=0)
        
        block_latents_new = buffer[begin_idx][None,] + cum_diff
        # print('block_latents_new shape', block_latents_new.shape)
        cum_error_vec = (block_latents_new - buffer[begin_idx + 1: end_idx +1])
        cur_error = (torch.linalg.norm(cum_error_vec.reshape(cum_error_vec.shape[0], cum_error_vec.shape[1], -1), dim=-1)).pow(2)
        # print('cur error', cur_error.shape)
        error_ratio = cur_error #* inv_sigmas.reshape(cur_error.size())
        # print('err_ratio', error_ratio)
        error_ratio = torch.nn.functional.pad(
            error_ratio, (0, 0, 0, 1), value=1e9
        )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
        any_error_at_time = torch.max(error_ratio > 0.01 ** 2, dim=1).values.int()
        ind = torch.argmax(any_error_at_time).item()
        #print(ind)
        #print(begin_idx, end_idx, stride)

        new_begin_idx = begin_idx + min(1 + ind, block_size)
        new_end_idx = min(new_begin_idx + block_size, sde.sample_N)
        #print(new_begin_idx, new_end_idx, stride)
        
        buffer[begin_idx + 1: end_idx + 1] = block_latents_new
        # linear extrap fill
        def get_deriv(b, e):
          deriv = (buffer[e] - buffer[b]) / (e - b)
          return deriv
      
        # deriv = (get_deriv(end_idx - 1, end_idx) + get_deriv(end_idx - 2, end_idx)) / 2
        deriv = (get_deriv(end_idx - 1, end_idx))
        buffer[end_idx: new_end_idx + 1] = buffer[end_idx][None,]
        if new_end_idx > end_idx:
          buffer[end_idx: new_end_idx + 1] += deriv * torch.arange(new_end_idx + 1 - end_idx)[:, None, None, None, None].to(device)
        
        begin_idx = new_begin_idx
        end_idx = new_end_idx
        
      x = inverse_scaler(buffer[-1])
      print("NFE", nfe_ctr)
      return x, nfe_ctr
 
  print('Type of Sampler:', sde.use_ode_sampler)
  if sde.use_ode_sampler=='rk45':
      return rk45_sampler
  elif sde.use_ode_sampler=='euler':
      # return euler_sampler
      return parallel_sampler
      # return strided_parallel_sampler
  else:
      assert False, 'Not Implemented!'
