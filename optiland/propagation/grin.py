# path: optiland/propagation/grin.py

"""
Implements the propagation model for Graded-Index (GRIN) media.

This module provides the GRINPropagation class, which conforms to the
BasePropagationModel interface. It encapsulates the logic for tracing rays
through a medium with a non-uniform refractive index by numerically solving
the Eikonal equation using a vectorized Runge-Kutta (RK4) method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import icontract

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel

# Use TYPE_CHECKING to avoid circular imports at runtime.
if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.materials.gradient_material import GradientMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.base import BaseSurface


class GRINPropagation(BasePropagationModel):
    """
    Handles ray propagation through a Graded-Index (GRIN) medium.

    This model uses a fourth-order Runge-Kutta (RK4) numerical integration
    scheme to solve the governing ray equation d/ds(n * dr/ds) = ∇n,
    simulating the curved path of light in a non-homogeneous medium.
    """

    @icontract.require(lambda step_size: step_size > 0)
    @icontract.require(lambda max_steps: max_steps > 0)
    def __init__(self, step_size: float = 0.1, max_steps: int = 10000) -> None:
        """
        Initializes the GRINPropagation model.

        Args:
            step_size: The fixed step size for each RK4 integration step (in mm).
            max_steps: The maximum number of integration steps to prevent
                       infinite loops.
        """
        self.step_size = step_size
        self.max_steps = max_steps

    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates rays from an entry surface to an exit surface through
        a GRIN medium.

        This method acts as the public interface. It validates that the
        propagation medium is a GradientMaterial and then delegates the
        complex numerical computation to a private solver method.
        """
        medium = surface_in.material_post
        
        # --- Contract Enforcement: Precondition Check ---
        # Ensure this propagation model is only used with the correct material type.
        if not hasattr(medium, 'get_index_and_gradient'):
             raise TypeError(
                "GRINPropagation can only be used with a material that has a "
                "'get_index_and_gradient' method, such as GradientMaterial."
             )

        # Delegate to the core solver.
        return self._solve_grin_path(
            rays_in,
            medium,
            surface_out,
            self.step_size,
            self.max_steps
        )

    def _solve_grin_path(
        self,
        rays_in: "RealRays",
        grin_material: "GradientMaterial",
        exit_surface: "BaseSurface",
        step_size: float,
        max_steps: int
    ) -> "RealRays":
        """
        Traces rays using RK4 integration until they intersect the exit surface.

        This is a vectorized implementation that processes a batch of rays
        simultaneously for performance. It tracks active rays and stops
        their propagation individually once they hit the boundary.
        """
        num_rays = len(rays_in.x)
        
        # Initialize state vectors for the integration.
        # r: position vector [x, y, z]
        # k: optical direction vector [n*L, n*M, n*N]
        r = be.stack([rays_in.x, rays_in.y, rays_in.z], axis=-1)
        d = be.stack([rays_in.L, rays_in.M, rays_in.N], axis=-1)
        
        wavelength = rays_in.w
        
        n_start, _ = grin_material.get_index_and_gradient(r[:, 0], r[:, 1], r[:, 2], wavelength)
        k = n_start[:, be.newaxis] * d
        
        opd = be.copy(rays_in.opd)
        active_rays = be.ones(num_rays, dtype=bool)

        # Pre-allocate arrays to store the final state of rays upon exit.
        final_r = be.copy(r)
        final_k = be.copy(k)
        
        def derivatives(current_r: Any, current_k: Any, w_active: Any) -> tuple:
            """Helper function to compute derivatives for the RK4 solver."""
            n, grad_n = grin_material.get_index_and_gradient(
                current_r[:, 0], current_r[:, 1], current_r[:, 2], w_active
            )
            # dr_ds = k / n
            dr_ds = current_k / (n[:, be.newaxis] + 1e-12) # Add epsilon for stability
            # dk_ds = ∇n
            dk_ds = grad_n
            return dr_ds, dk_ds

        for _ in range(max_steps):
            if not be.any(active_rays):
                break

            # Select only the active rays for this iteration's computation.
            r_active = r[active_rays]
            k_active = k[active_rays]
            w_active = wavelength if be.ndim(wavelength) == 0 else wavelength[active_rays]
            
            n_current, _ = grin_material.get_index_and_gradient(
                r_active[:, 0], r_active[:, 1], r_active[:, 2], w_active
            )

            # --- RK4 Integration Step ---
            r1, k1 = derivatives(r_active, k_active, w_active)
            r2, k2 = derivatives(r_active + 0.5 * step_size * r1, k_active + 0.5 * step_size * k1, w_active)
            r3, k3 = derivatives(r_active + 0.5 * step_size * r2, k_active + 0.5 * step_size * k2, w_active)
            r4, k4 = derivatives(r_active + step_size * r3, k_active + step_size * k3, w_active)

            r_next_active = r_active + (step_size / 6.0) * (r1 + 2*r2 + 2*r3 + r4)
            k_next_active = k_active + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Update the main state array for all active rays.
            r[active_rays] = r_next_active
            k[active_rays] = k_next_active
            
            # Accumulate Optical Path Difference (OPD).
            n_next, _ = grin_material.get_index_and_gradient(
                r_next_active[:, 0], r_next_active[:, 1], r_next_active[:, 2], w_active
            )
            opd[active_rays] += 0.5 * (n_current + n_next) * step_size

            # --- Intersection Check ---
            # Check for intersection within the segment [r_active, r_next_active].
            segment_vec = r_next_active - r_active
            segment_len = be.linalg.norm(segment_vec, axis=-1)
            
            safe_segment_len = segment_len + 1e-12
            segment_dir = segment_vec / safe_segment_len[:, be.newaxis]

            # Use a temporary RealRays object for the intersection test.
            segment_rays = RealRays(
                x=r_active[:, 0], y=r_active[:, 1], z=r_active[:, 2],
                L=segment_dir[:, 0], M=segment_dir[:, 1], N=segment_dir[:, 2],
                intensity=be.ones_like(segment_len), wavelength=w_active
            )
            
            # The intersection test requires localization to the surface's frame.
            exit_surface.geometry.localize(segment_rays)
            distance_to_intersect = exit_surface.geometry.distance(segment_rays)
            
            # Identify rays that intersected within the current step.
            intersected_mask_local = (distance_to_intersect > 1e-9) & (distance_to_intersect <= segment_len)

            if be.any(intersected_mask_local):
                # Get the global indices of rays that just finished propagating.
                active_indices_global = be.where(active_rays)[0]
                intersected_indices_global = active_indices_global[intersected_mask_local]

                # Calculate exact intersection point.
                intersection_point = r_active[intersected_mask_local] + \
                    distance_to_intersect[intersected_mask_local, be.newaxis] * segment_dir[intersected_mask_local]
                
                # Store their final state and deactivate them.
                final_r[intersected_indices_global] = intersection_point
                final_k[intersected_indices_global] = k_next_active[intersected_mask_local]
                active_rays[intersected_indices_global] = False
        else: # This 'else' belongs to the 'for' loop, executing if the loop completes without 'break'.
            if be.any(active_rays):
                # If loop finishes and rays are still active, they failed to intersect.
                raise ValueError("Some rays did not intersect the exit surface after the maximum number of steps.")

        # --- Finalization ---
        # Construct the final RealRays object from the stored final states.
        w_final = rays_in.w
        n_final, _ = grin_material.get_index_and_gradient(final_r[:, 0], final_r[:, 1], final_r[:, 2], w_final)
        final_d = final_k / (n_final[:, be.newaxis] + 1e-12)
        
        # Re-normalize direction cosines for safety.
        final_d_norm = be.linalg.norm(final_d, axis=-1)
        final_d /= (final_d_norm[:, be.newaxis] + 1e-12)

        rays_out = RealRays(
            x=final_r[:, 0], y=final_r[:, 1], z=final_r[:, 2],
            L=final_d[:, 0], M=final_d[:, 1], N=final_d[:, 2],
            intensity=be.copy(rays_in.i), wavelength=be.copy(rays_in.w)
        )
        rays_out.opd = opd
        
        return rays_out

    @classmethod
    def from_dict(cls, d: dict, material: "BaseMaterial" = None) -> "GRINPropagation":
        """
        Creates a GRINPropagation model from a dictionary.

        This factory method supports deserialization, allowing configurable
        parameters like step_size and max_steps to be specified in the
        serialized format.

        Args:
            d: The dictionary representation of the model.
            material: The parent material instance. This is accepted for
                API consistency but is not used by this model.

        Returns:
            An instance of the GRINPropagation model.
        """
        # Extract configuration from dict, with defaults.
        step_size = d.get('step_size', 0.1)
        max_steps = d.get('max_steps', 10000)
        return cls(step_size=step_size, max_steps=max_steps)