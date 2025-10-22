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
from optiland.rays import RealRays

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.surfaces.base import BaseSurface


class HomogeneousPropagation(BasePropagationModel):
    """
    Handles ray propagation in a straight line through a homogeneous,
    isotropic medium.
    """

    def __init__(self, material: "BaseMaterial" | None = None) -> None:
        """
        Initializes the HomogeneousPropagation model.
        
        Note: The `material` argument is kept for backward compatibility with the
        factory's `from_dict` method signature but is not used by this model's
        `propagate` method, which instead derives the medium from `surface_in`.
        """
        pass

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
        calculations.

        The process is as follows:
        1. Create a deep copy of the input rays (`rays_out`) to ensure immutability.
        2. Create a second temporary copy (`rays_for_intersect`) for calculation.
        3. Localize `rays_for_intersect` into the coordinate frame of the exit surface.
        4. Calculate the geometric distance to the exit surface using the localized rays.
        5. Update the `rays_out` object's global coordinates and Optical Path Difference.
        """
        # --- Contract Enforcement: Immutability ---
        # Create the final output object by copying all attributes from the input.
        rays_out = RealRays(
            x=be.copy(rays_in.x),
            y=be.copy(rays_in.y),
            z=be.copy(rays_in.z),
            L=be.copy(rays_in.L),
            M=be.copy(rays_in.M),
            N=be.copy(rays_in.N),
            intensity=be.copy(rays_in.i),
            wavelength=be.copy(rays_in.w)
        )
        rays_out.opd = be.copy(rays_in.opd)
        
        # --- Coordinate System Handling for Intersection ---
        # Create a temporary ray object to be transformed for the distance calculation.
        # This preserves the global coordinates of rays_out for the final update.
        rays_for_intersect = RealRays(
            x=be.copy(rays_in.x), y=be.copy(rays_in.y), z=be.copy(rays_in.z),
            L=be.copy(rays_in.L), M=be.copy(rays_in.M), N=be.copy(rays_in.N),
            intensity=be.copy(rays_in.i), wavelength=be.copy(rays_in.w)
        )
        
        # Localize the temporary rays into the exit surface's coordinate frame.
        surface_out.geometry.localize(rays_for_intersect)

        # --- Core Logic ---
        # 1. Calculate the geometric distance. This is now correct because it operates
        #    on rays represented in the surface's local coordinate system.
        distance = surface_out.geometry.distance(rays_for_intersect)
        
        # 2. Update the final output ray's positions (which are in global coords).
        rays_out.x += distance * rays_out.L
        rays_out.y += distance * rays_out.M
        rays_out.z += distance * rays_out.N

        # 3. Update the Optical Path Difference.
        medium = surface_in.material_post
        n = medium.n(rays_out.w)
        rays_out.opd += n * distance

        return rays_out

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "HomogeneousPropagation":
        """
        Creates a HomogeneousPropagation model from a dictionary.
        
        Args:
            d: The dictionary representation of the model (often just `{'class': ...}`).
            material: The parent material instance, required by the base class factory.

        Returns:
            An instance of the HomogeneousPropagation model.
        """
        return cls(material=material)