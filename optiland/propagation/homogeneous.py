# path: optiland/propagation/homogeneous.py

"""
Implements the standard straight-line propagation model for homogeneous media.

This module provides the default propagation behavior for rays traveling through
a medium with a uniform refractive index, encapsulating the logic previously
hardcoded in the main trace loop.
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
    isotropic medium.
    """

    def __init__(self, material: "BaseMaterial" | None = None) -> None:
        """
        Initializes the HomogeneousPropagation model.
        
        Args:
            material: The parent material instance. This reference is stored
                      to support serialization and introspection patterns.
        """
        self.material = material

    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates rays from an entry surface to an exit surface in a straight line.

        This implementation adheres to the functional contract of immutability and
        correctly handles coordinate system transformations for intersection
        calculations. It also applies attenuation based on the Beer-Lambert law
        and ensures output rays are normalized.
        """
        # --- Contract Enforcement: Immutability ---
        # Create a deep copy of the input rays to ensure the original is not modified.
        rays_out = rays_in.copy()
        
        # --- Core Logic ---
        # 1. Calculate the geometric distance to the exit surface.
        #    The surface's geometry object handles any necessary coordinate transforms.
        #    We must localize rays to the exit surface's coordinate system first.
        surface_out.geometry.localize(rays_out)
        distance = surface_out.geometry.distance(rays_out)
        surface_out.geometry.globalize(rays_out) # Return rays to global system

        # 2. Update the final output ray's positions.
        rays_out.x += distance * rays_out.L
        rays_out.y += distance * rays_out.M
        rays_out.z += distance * rays_out.N

        # 3. Handle physical effects within the medium.
        medium = surface_in.material_post
        
        # 3a. Update the Optical Path Difference.
        n = medium.n(rays_out.w)
        rays_out.opd += n * distance
        
        # 3b. Apply attenuation based on Beer-Lambert law if k > 0.
        k = medium.k(rays_out.w)
        # alpha = 4 * pi * k / lambda
        # I = I_0 * exp(-alpha * z)
        # Note: distance is in mm, wavelength w is in um. Convert distance to um.
        alpha = 4 * be.pi * k / (rays_out.w + 1e-12)
        attenuation_factor = be.exp(-alpha * distance * 1e3)
        rays_out.i *= attenuation_factor

        # 4. Ensure final direction cosines are normalized.
        rays_out.normalize()

        return rays_out

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "HomogeneousPropagation":
        """
        Creates a HomogeneousPropagation model from a dictionary.
        
        Args:
            d: The dictionary representation of the model.
            material: The parent material instance, required by the base class factory.

        Returns:
            An instance of the HomogeneousPropagation model.
        """
        return cls(material=material)