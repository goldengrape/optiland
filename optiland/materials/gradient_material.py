"""Defines a gradient-index material and the calculation of its physical properties."""

from dataclasses import dataclass, field
import icontract
import numpy as np
from typing import Tuple

from optiland.materials.base import BaseMaterial

@icontract.invariant(
    lambda self: all(isinstance(getattr(self, c), (int, float)) for c in self.__annotations__ if c != 'name'),
    "All refractive index coefficients must be numeric types"
)
@dataclass(frozen=True)
class GradientMaterial(BaseMaterial):
    """
    A gradient-index material defined by a polynomial.

    The refractive index n is calculated as:
    n(r, z) = n0 + nr2*r^2 + nr4*r^4 + nr6*r^6 + nz1*z + nz2*z^2 + nz3*z^3
    where r^2 = x^2 + y^2.

    All coefficients are treated as immutable to encourage a functional programming style.
    """
    n0: float = 1.0
    nr2: float = 0.0
    nr4: float = 0.0
    nr6: float = 0.0
    nz1: float = 0.0
    nz2: float = 0.0
    nz3: float = 0.0
    name: str = "GRIN Material"

    @icontract.require(lambda x, y, z: all(isinstance(v, (int, float, np.ndarray)) for v in [x, y, z]))
    def get_index(self, x: float, y: float, z: float) -> float:
        """
        Calculates the refractive index n at a given coordinate (x, y, z). This is a pure function.
        """
        r2 = x**2 + y**2
        n = (self.n0 +
             self.nr2 * r2 +
             self.nr4 * r2**2 +
             self.nr6 * r2**3 +
             self.nz1 * z +
             self.nz2 * z**2 +
             self.nz3 * z**3)
        return float(n)

    @icontract.require(lambda x, y, z: all(isinstance(v, (int, float, np.ndarray)) for v in [x, y, z]))
    @icontract.ensure(lambda result: result.shape == (3,))
    def get_gradient(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculates the gradient of the refractive index ∇n = [∂n/∂x, ∂n/∂y, ∂n/∂z]
        at a given coordinate (x, y, z). This is a pure function.
        """
        r2 = x**2 + y**2
        dn_dr2 = self.nr2 + 2 * self.nr4 * r2 + 3 * self.nr6 * r2**2
        dn_dx = 2 * x * dn_dr2
        dn_dy = 2 * y * dn_dr2
        dn_dz = self.nz1 + 2 * self.nz2 * z + 3 * self.nz3 * z**2
        return np.array([dn_dx, dn_dy, dn_dz], dtype=float)

    def get_index_and_gradient(self, x: float, y: float, z: float) -> Tuple[float, np.ndarray]:
        """
        Calculates both the refractive index n and its gradient ∇n in a single call
        for performance optimization.
        """
        r2 = x**2 + y**2
        n = (self.n0 +
             self.nr2 * r2 +
             self.nr4 * r2**2 +
             self.nr6 * r2**3 +
             self.nz1 * z +
             self.nz2 * z**2 +
             self.nz3 * z**3)

        dn_dr2 = self.nr2 + 2 * self.nr4 * r2 + 3 * self.nr6 * r2**2
        dn_dx = 2 * x * dn_dr2
        dn_dy = 2 * y * dn_dr2
        dn_dz = self.nz1 + 2 * self.nz2 * z + 3 * self.nz3 * z**2

        return float(n), np.array([dn_dx, dn_dy, dn_dz], dtype=float)

    def _calculate_n(self, wavelength, **kwargs):
        if 'x' in kwargs and 'y' in kwargs and 'z' in kwargs:
            return self.get_index(kwargs['x'], kwargs['y'], kwargs['z'])
        else:
            return self.n0

    def _calculate_k(self, wavelength, **kwargs):
        return 0.0
