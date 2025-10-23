# path: optiland/propagation/homogeneous.py

"""
Implements the standard straight-line propagation model for homogeneous media.

This module provides the default propagation behavior for rays traveling through
a medium with a uniform refractive index. This implementation favors performance
by modifying the input rays object in-place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel

# Use TYPE_CHECKING to avoid circular imports at runtime.
if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.base import BaseSurface


class HomogeneousPropagation(BasePropagationModel):
    """
    Handles ray propagation in a straight line through a homogeneous,
    isotropic medium by modifying the ray data in-place for maximum performance.
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
        Propagates rays from an entry to an exit surface by modifying the
        `rays_in` object directly.

        This method follows a performance-oriented, imperative contract. The state
        of the `rays_in` object is updated to reflect its new position and
        properties at the exit surface.

        Returns:
            The modified `rays_in` object itself, allowing for method chaining.
        """
        # --- High-Performance Contract: In-place Modification ---
        # The 'rays_in' object will be mutated directly. No copy is made.
        
        # 1. Calculate the geometric distance to the exit surface.
        #    We must localize rays to the exit surface's coordinate system first.
        surface_out.geometry.localize(rays_in)
        distance = surface_out.geometry.distance(rays_in)
        surface_out.geometry.globalize(rays_in) # Return rays to global system

        # 2. Update the ray positions.
        rays_in.x += distance * rays_in.L
        rays_in.y += distance * rays_in.M
        rays_in.z += distance * rays_in.N

        # 3. Handle physical effects within the medium.
        medium = surface_in.material_post
        
        # 3a. Update the Optical Path Difference.
        n = medium.n(rays_in.w)
        rays_in.opd += n * distance
        
        # 3b. Apply attenuation based on Beer-Lambert law if k > 0.
        k = medium.k(rays_in.w)
        # alpha = 4 * pi * k / lambda
        # I = I_0 * exp(-alpha * z)
        # Note: distance is in mm, wavelength w is in um. Convert distance to um.
        alpha = 4 * be.pi * k / (rays_in.w + 1e-12)
        attenuation_factor = be.exp(-alpha * distance * 1e3)
        rays_in.i *= attenuation_factor

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