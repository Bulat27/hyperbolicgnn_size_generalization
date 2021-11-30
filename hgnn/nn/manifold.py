import abc
import torch
from util import dot

EPS = 1e-5

class Manifold(abc):
    r""" Abstract base class for a Manifold.
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def log(self, y, x=None):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def exp(self, v, x=None):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def parallel_transport(self, v, x):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def dist(self, x, y):
        return NotImplementedError


class EuclideanManifold(Manifold):

    def __init__(self):
        super().__init__('Euclidean')

    @property
    def log(self, y, x=None):
        if x is None:
            x = torch.zeros_like(y)
        return y - x

    @property
    def exp(self, v, x=None):
        if x is None:
            x = torch.zeros_like(v)
        return v + x

    @property
    def parallel_transport(self, v, x):
        return v - x

    @property
    def dist(self, x, y):
        return torch.linalg.norm(x - y, dim=-1)


class PoincareBallManifold(Manifold):

    def __init__(self):
        super().__init__('Poincare Ball')

    def mobius_add(self, x, y):
        x_dot_y = dot(x, y)
        x_norm_squared = dot(x, x)
        y_norm_squared = dot(y, y)
        numerator = (1 + 2 * x_dot_y + y_norm_squared) * x + (1 - x_norm_squared) * y
        denominator = 1 + 2 * x_dot_y + x_norm_squared * y_norm_squared
        return numerator / denominator

    @property
    def log(self, y, x=None):
        if x is None:
            x = torch.zeros_like(y)
        norm_y = torch.linalg.norm(y, dim=-1, keepdim=True)
        return torch.arctanh(norm_y) * (y / norm_y.clip(EPS))

    @property
    def exp(self, v, x=None):
        if x is None:
            x = torch.zeros_like(v)
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        return torch.tanh(norm_v) * (v / norm_v.clip(EPS))

    @property
    def parallel_transport(self, v, x):
        return NotImplementedError

    @property
    def dist(self, x, y):
        x_y = x - y
        return torch.arccosh(1 + 2 * (dot(x_y, x_y) / ((1 - dot(x, x)) * (1 - dot(y, y)))))


class LorentzManifold(Manifold):

    def __init__(self):
        super().__init__('Lorentz')

    def scalar_product(self, x, y, keepdim=True):
        res = -(x[..., 0:1] * y[..., 0:1]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        return res if keepdim else res.squeeze(-1)

    @property
    def log(self, y, x=None):
        if x is None:
            x = torch.zeros_like(y)
            x[..., 0] = 1
        return (self.dist(x, y) / torch.sqrt(self.scalar_product(x, y).pow(2) - 1).clip(EPS)) * (y + self.scalar_product(x, y) * x)

    @property
    def exp(self, v, x=None):
        if x is None:
            x = torch.zeros_like(v)
            x[..., 0] = 1
        lorentz_norm_v = torch.sqrt(self.scalar_product(v, v))
        return torch.cosh(lorentz_norm_v) * x + torch.sinh(lorentz_norm_v) * (v / lorentz_norm_v.clip(EPS))

    @property
    def parallel_transport(self, v, x):
        return NotImplementedError

    @property
    def dist(self, x, y):
        return torch.arccosh(-self.scalar_product(x, y))