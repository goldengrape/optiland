# path: optiland/propagation/homogeneous.py

"""
Implements the standard straight-line propagation model for homogeneous media.

This module provides the default propagation behavior for rays traveling through
a medium with a uniform refractive index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.base import BaseSurface


class HomogeneousPropagation(BasePropagationModel):
    """
    Handles ray propagation in a straight line through a homogeneous,
    isotropic medium.
    """

    def __init__(self, material: "BaseMaterial" | None = None) -> None:
        """
        Initializes the HomogeneousPropagation model.
        
        Args:
            material: The parent material instance.
        """
        self.material = material

    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates rays from an entry to an exit surface.

        This method follows a functional contract at the object level: it mutates
        the attributes of the `rays_in` object and returns it. To support
        differentiable backends like PyTorch, tensor modifications are performed
        out-of-place (e.g., `x = x + delta`) rather than in-place (`x += delta`).

        Returns:
            The modified `rays_in` object itself, allowing for method chaining.
        """
        # 1. Calculate the geometric distance to the exit surface.
        surface_out.geometry.localize(rays_in)
        distance = surface_out.geometry.distance(rays_in)
        surface_out.geometry.globalize(rays_in) # Return rays to global system

        # FIX: Use out-of-place assignments instead of in-place operations (+=, *=)
        # to ensure compatibility with PyTorch's autograd engine.
        
        # 2. Update the ray positions.
        rays_in.x = rays_in.x + distance * rays_in.L
        rays_in.y = rays_in.y + distance * rays_in.M
        rays_in.z = rays_in.z + distance * rays_in.N

        # 3. Handle physical effects within the medium.
        medium = surface_in.material_post
        
        # 3a. Update the Optical Path Difference.
        n = medium.n(rays_in.w)
        rays_in.opd = rays_in.opd + n * distance
        
        # 3b. Apply attenuation based on Beer-Lambert law if k > 0.
        k = medium.k(rays_in.w)
        # alpha = 4 * pi * k / lambda
        # I = I_0 * exp(-alpha * z)
        # Note: distance is in mm, wavelength w is in um. Convert distance to um.
        alpha = 4 * be.pi * k / (rays_in.w + 1e-12)
        attenuation_factor = be.exp(-alpha * distance * 1e3)
        rays_in.i = rays_in.i * attenuation_factor

        # 4. Ensure final direction cosines are normalized.
        rays_in.normalize()

        return rays_in

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "HomogeneousPropagation":
        """
        Creates a HomogeneousPropagation model from a dictionary.
        """
        return cls(material=material)