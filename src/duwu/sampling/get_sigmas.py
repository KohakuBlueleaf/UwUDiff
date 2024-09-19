from collections.abc import Callable

import numpy as np


def get_sigmas_for_rf(
    num_steps, max_sigma, min_sigma=0, time_disc_func: Callable | None = None
):
    max_time = max_sigma / (1 + max_sigma)
    min_time = min_sigma / (1 + min_sigma)
    time_disc_func = time_disc_func or uniform_time
    time = np.flip(time_disc_func(min_time, max_time, num_steps))
    sigmas = time / (1 - time)
    return sigmas


def uniform_time(min_time, max_time, num_steps):
    return np.linspace(min_time, max_time, num_steps + 1)


def sigmoid_time(min_time, max_time, num_steps, rho=10):
    # independent of rho
    min_time = max(min_time, 1e-5)
    min_time_logit = np.log(min_time / (1 - min_time))
    max_time_logit = np.log(max_time / (1 - max_time))
    min_time_rt = min_time_logit / rho + 0.5
    max_time_rt = max_time_logit / rho + 0.5
    time_rt = np.linspace(min_time_rt, max_time_rt, num_steps + 1)
    time = 1 / (1 + np.exp(-rho * (time_rt - 0.5)))
    time[0] = min_time
    return time


def sigmoid_time_scale(min_time, max_time, num_steps, rho=10):
    time_rt = np.linspace(-0.5, 0.5, num_steps + 1)
    time = 1 / (1 + np.exp(-rho * time_rt))
    # scale to [0, 1]
    time = (time - time[0]) / (time[-1] - time[0])
    # scale to [min_time, max_time]
    time = time * (max_time - min_time) + min_time
    return time
