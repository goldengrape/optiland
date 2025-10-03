"""
Implements the ray propagation algorithm in a Gradient Refractive Index (GRIN) medium.
It uses a vectorized RK4 numerical integration method to solve the ray equation: d/ds(n * dr/ds) = ∇n
"""
import icontract
from typing import Tuple

import optiland.backend as be
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface as BaseSurface
from optiland.materials.gradient_material import GradientMaterial


@icontract.require(lambda rays_in: isinstance(rays_in, RealRays))
@icontract.require(lambda step_size: step_size > 0)
@icontract.require(lambda max_steps: max_steps > 0)
def propagate_through_gradient(
    rays_in: RealRays,
    grin_material: "GradientMaterial",
    exit_surface: "BaseSurface",
    step_size: float = 0.1,
    max_steps: int = 10000
) -> RealRays:
    """
    Traces a batch of rays through a GRIN medium until they intersect the exit surface.

    Args:
        rays_in: The initial state of the rays (positions and directions).
        grin_material: The physical model of the GRIN medium.
        exit_surface: The geometric surface marking the end of the GRIN medium.
        step_size: The step size for RK4 integration (in mm).
        max_steps: The maximum number of steps to prevent infinite loops.

    Returns:
        The final state of the rays at the exit surface.
    """
    # Extract initial positions and directions
    r = be.stack([rays_in.x, rays_in.y, rays_in.z], axis=-1)
    d = be.stack([rays_in.L, rays_in.M, rays_in.N], axis=-1)

    # Initialize optical momentum k = n * d
    n_start, _ = grin_material.get_index_and_gradient(r[:, 0], r[:, 1], r[:, 2])
    k = n_start[:, be.newaxis] * d
    
    # Keep track of optical path difference for each ray
    opd = be.copy(rays_in.opd)

    num_rays = len(rays_in.x)
    active_rays = be.ones(num_rays, dtype=bool)
    
    # Store final state of the rays that have exited
    final_r = be.copy(r)
    final_k = be.copy(k)

    def derivatives(current_r: be.ndarray, current_k: be.ndarray) -> tuple:
        n, grad_n = grin_material.get_index_and_gradient(current_r[:, 0], current_r[:, 1], current_r[:, 2])
        # Add a small epsilon to n to avoid division by zero
        dr_ds = current_k / (n[:, be.newaxis] + 1e-9)
        dk_ds = grad_n
        return dr_ds, dk_ds

    for i in range(max_steps):
        if not be.any(active_rays):
            break

        r_active = r[active_rays]
        k_active = k[active_rays]

        n_current = grin_material.get_index(r_active[:, 0], r_active[:, 1], r_active[:, 2])

        # RK4 integration step for active rays
        r1, k1 = derivatives(r_active, k_active)
        r2, k2 = derivatives(r_active + 0.5 * step_size * r1, k_active + 0.5 * step_size * k1)
        r3, k3 = derivatives(r_active + 0.5 * step_size * r2, k_active + 0.5 * step_size * k2)
        r4, k4 = derivatives(r_active + step_size * r3, k_active + step_size * k3)

        r_next_active = r_active + (step_size / 6.0) * (r1 + 2*r2 + 2*r3 + r4)
        k_next_active = k_active + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Accumulate Optical Path Difference (OPD)
        n_next = grin_material.get_index(r_next_active[:, 0], r_next_active[:, 1], r_next_active[:, 2])
        opd[active_rays] += 0.5 * (n_current + n_next) * step_size

        # Check for intersection with the exit surface
        segment_vec = r_next_active - r_active
        segment_len = be.linalg.norm(segment_vec, axis=-1)
        
        # Create a RealRays object for the current segment
        # Note: This is a bit inefficient, but exit_surface.intersect expects RealRays
        segment_rays = RealRays(
            x=r_active[:, 0], y=r_active[:, 1], z=r_active[:, 2],
            L=segment_vec[:, 0] / (segment_len[:, be.newaxis] + 1e-9), 
            M=segment_vec[:, 1] / (segment_len[:, be.newaxis] + 1e-9), 
            N=segment_vec[:, 2] / (segment_len[:, be.newaxis] + 1e-9),
            intensity=be.ones_like(segment_len), wavelength=rays_in.w[active_rays]
        )

        # Localize rays to the exit surface's coordinate system
        exit_surface.geometry.localize(segment_rays)

        distance_to_intersect = exit_surface.geometry.distance(segment_rays)

        # Globalize rays back
        exit_surface.geometry.globalize(segment_rays)

        # Identify rays that intersect within the current step
        intersected_mask = (distance_to_intersect > 1e-9) & (distance_to_intersect <= segment_len)

        if be.any(intersected_mask):
            # Get indices of the rays that have intersected
            intersected_indices_local = be.where(intersected_mask)[0]
            intersected_indices_global = be.where(active_rays)[0][intersected_indices_local]

            # Calculate exact intersection point and update final state
            intersection_point = r_active[intersected_mask] + distance_to_intersect[intersected_mask, be.newaxis] * be.stack([segment_rays.L[intersected_mask], segment_rays.M[intersected_mask], segment_rays.N[intersected_mask]], axis=-1)
            final_r[intersected_indices_global] = intersection_point
            final_k[intersected_indices_global] = k_next_active[intersected_mask]
            
            # Deactivate these rays
            active_rays[intersected_indices_global] = False

        # Update state for non-intersected (still active) rays
        still_active_local_mask = ~intersected_mask
        
        # Get the global indices of the rays that were active in this step
        active_indices_global = be.where(active_rays)[0]
        # Among those, find the ones that did *not* intersect
        still_active_global_indices = active_indices_global[still_active_local_mask]
        
        # Create a new mask for the next iteration
        next_active_rays = be.zeros(num_rays, dtype=bool)
        if len(still_active_global_indices) > 0:
            next_active_rays[still_active_global_indices] = True
        
        # Update the r and k arrays only for the rays that are still active
        r_still_active = r_next_active[still_active_local_mask]
        k_still_active = k_next_active[still_active_local_mask]

        # The r and k arrays for the next loop should only contain the still active rays
        # We can just update the main r and k arrays at the correct indices
        if be.any(next_active_rays):
            r[next_active_rays] = r_still_active
            k[next_active_rays] = k_still_active
        
        active_rays = next_active_rays

    if be.any(active_rays):
        raise ValueError("Some rays did not intersect the exit surface after the maximum number of steps.")

    # Finalize the output rays
    n_final = grin_material.get_index(final_r[:, 0], final_r[:, 1], final_r[:, 2])
    final_d = final_k / (n_final[:, be.newaxis] + 1e-9)
    final_d_norm = be.linalg.norm(final_d, axis=-1)
    final_d /= final_d_norm[:, be.newaxis]

    rays_out = RealRays(
        x=final_r[:, 0], y=final_r[:, 1], z=final_r[:, 2],
        L=final_d[:, 0], M=final_d[:, 1], N=final_d[:, 2],
        intensity=rays_in.i, wavelength=rays_in.w
    )
    rays_out.opd = opd
    return rays_out
