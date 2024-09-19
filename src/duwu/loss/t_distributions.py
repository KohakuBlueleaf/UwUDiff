from abc import abstractmethod

import numpy as np
import torch


def get_t_dist(cfg):
    if cfg["name"] == "uniform":
        return Uniform(ub=cfg.get("ub", 1.0), lb=cfg.get("lb", 0.0))
    elif cfg["name"] == "lognormal":
        return LogNormal(mu=cfg.get("mu", -1.2), std=cfg.get("std", 1.2))
    elif cfg["name"] == "exp":
        return Exp(rho_1=cfg.get("rho_1", 4.0))
    elif cfg["name"] == "cosh":
        return Cosh(a=cfg.get("a", 4.0))
    elif cfg["name"] == "saw":
        return Saw(a=cfg.get("a", 4), rho_1=cfg.get("rho_1", 1.0))
    else:
        return None


class Distribution:

    @abstractmethod
    def __updf__(self, t):
        pass

    @property
    def __updf_Z__(self):
        return self.__updf__(torch.linspace(0, 1, 2000)[1:]).mean()

    def pdf(self, t):
        return self.__updf__(t) / self.__updf_Z__.to(t.device)

    def kappa(self, t):
        pdf_min = self.pdf(torch.linspace(0, 1, 2000)[1:]).min().to(t.device)
        pdf = self.pdf(t)
        return (pdf / pdf_min).clamp(max=100)
        # return (pdf.max()/pdf.min()).clamp(max=100).item()

    def sample(self, bs):
        ls = torch.linspace(0, 1, 2000)[1:]
        updf = self.__updf__(ls).reshape(1, -1).repeat(bs, 1)
        return ls[torch.multinomial(updf, 1).reshape(-1)]


class Uniform(Distribution):
    def __init__(self, ub, lb):
        self.ub = ub
        self.lb = lb

    def __updf__(self, t):
        updf = torch.zeros_like(t)
        updf[(t >= self.lb) * (t <= self.ub)] = 1.0
        return updf


class LogNormal(Distribution):
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __updf__(self, t):
        s = t / (1 - t)
        updf = (
            (-(s.log() - self.mu).square() / (2 * self.std**2)).exp()
            / (s * self.std)
            / (1 - t).square()
        )
        updf[(t == 1.0)] = 0.0
        return updf


class Exp(Distribution):
    def __init__(self, rho_1):
        self.rho_1 = float(rho_1)

    def __updf__(self, t):
        return (np.log(self.rho_1) * (t - 0.5)).exp()


class Cosh(Distribution):
    def __init__(self, a):
        self.a = a

    def __updf__(self, t):
        return torch.cosh(self.a * (t - 0.5))


class Saw(Distribution):
    def __init__(self, a, rho_1):
        self.a = a
        self.rho_1 = rho_1

    def __updf__(self, t):
        p1 = torch.exp(self.a * t)
        p0 = torch.exp(self.a * (t + 0.6))
        p0[t > 0.4] = 0.0
        return p0 + self.rho_1 * p1
