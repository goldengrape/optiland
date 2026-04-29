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
        surface_in_or_distance: "BaseSurface | Any",
        surface_out: "BaseSurface | None" = None,
    ) -> "RealRays":
        """
        Propagates rays through a homogeneous medium.

        The method accepts both the upstream distance-based contract
        ``propagate(rays, t)`` and the GRIN-oriented surface contract
        ``propagate(rays, surface_in, surface_out)``. In both cases it mutates
        and returns ``rays_in``.

        Returns:
            The modified `rays_in` object itself, allowing for method chaining.
        """
        if surface_out is None:
            distance = surface_in_or_distance
            medium = self.material
            update_opd = False
            force_normalize = False
        else:
            surface_in = surface_in_or_distance
            medium = surface_in.material_post
            update_opd = True
            force_normalize = True

            # Calculate the geometric distance to the exit surface.
            surface_out.geometry.localize(rays_in)
            distance = surface_out.geometry.distance(rays_in)
            surface_out.geometry.globalize(rays_in)

        self._propagate_distance(rays_in, distance, medium, update_opd, force_normalize)
        return rays_in

    def _propagate_distance(
        self,
        rays_in: "RealRays",
        distance: Any,
        medium: "BaseMaterial",
        update_opd: bool,
        force_normalize: bool,
    ) -> None:
        """Propagate rays in-place by a known geometric distance."""
        if medium is None:
            raise ValueError("HomogeneousPropagation requires a material.")

        # Use out-of-place assignments for PyTorch autograd compatibility.
        rays_in.x = rays_in.x + distance * rays_in.L
        rays_in.y = rays_in.y + distance * rays_in.M
        rays_in.z = rays_in.z + distance * rays_in.N

        if update_opd:
            rays_in.opd = rays_in.opd + medium.n(rays_in.w) * distance

        k = medium.k(rays_in.w)
        if be.any(k > 0):
            alpha = 4 * be.pi * k / (rays_in.w + 1e-12)
            rays_in.i = rays_in.i * be.exp(-alpha * distance * 1e3)

        if force_normalize or not rays_in.is_normalized:
            rays_in.normalize()

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "HomogeneousPropagation":
        """
        Creates a HomogeneousPropagation model from a dictionary.
        """
        return cls(material=material)
