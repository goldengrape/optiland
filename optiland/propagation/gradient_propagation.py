# path: optiland/propagation/gradient_propagation.py

"""
Implements the core ray propagation algorithm in a Gradient Refractive Index (GRIN) medium.

This module provides a pure, vectorized function that uses the RK4 numerical 
integration method to solve the ray equation: d/ds(n * dr/ds) = ∇n. It is designed
to be called by propagation models and is decoupled from the model interface itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import icontract

import optiland.backend as be
from optiland.rays import RealRays

if TYPE_CHECKING:
    from optiland.materials.gradient_material import GradientMaterial
    from optiland.surfaces.base import BaseSurface


@icontract.require(lambda rays_in: isinstance(rays_in, RealRays))
@icontract.require(lambda step_size: step_size > 0)
@icontract.require(lambda max_steps: max_steps > 0)
def propagate_through_gradient(
    rays_in: RealRays,
    grin_material: "GradientMaterial",
    exit_surface: "BaseSurface",
    step_size: float,
    max_steps: int
) -> RealRays:
    """
    Traces a batch of rays using RK4 integration until they intersect the exit surface.

    This is a vectorized implementation that processes a batch of rays
    simultaneously for performance. It tracks active rays and stops
    their propagation individually once they hit the boundary.

    Args:
        rays_in: The initial state of the rays (positions and directions).
        grin_material: The physical model of the GRIN medium.
        exit_surface: The geometric surface marking the end of the GRIN medium.
        step_size: The step size for RK4 integration (in mm).
        max_steps: The maximum number of steps to prevent infinite loops.

    Returns:
        A new `RealRays` object representing the final state of the rays at the exit surface.
    """
    num_rays = len(rays_in.x)
    
    # Initialize state vectors for the integration.
    # r: position vector [x, y, z]
    # k: optical direction vector [n*L, n*M, n*N]
    r = be.stack([rays_in.x, rays_in.y, rays_in.z], axis=-1)
    d = be.stack([rays_in.L, rays_in.M, rays_in.N], axis=-1)
    
    wavelength = rays_in.w
    
    n_start, _ = grin_material.get_index_and_gradient(r[:, 0], r[:, 1], r[:, 2], wavelength)
    k = be.unsqueeze_last(n_start) * d
    
    opd = be.copy(rays_in.opd)
    
    # FIX: Create a boolean tensor/array in a backend-agnostic way.
    # be.ones((num_rays,)) creates a float array of [1., 1., ...].
    # Comparing it to 0 yields a boolean array of [True, True, ...].
    # The shape is passed as a tuple to satisfy PyTorch's `ones` signature.
    active_rays = be.ones((num_rays,)) > 0

    # Pre-allocate arrays to store the final state of rays upon exit.
    final_r = be.copy(r)
    final_k = be.copy(k)
    
    def derivatives(current_r: Any, current_k: Any, w_active: Any) -> tuple[Any, Any]:
        """Helper function to compute derivatives for the RK4 solver."""
        n, grad_n = grin_material.get_index_and_gradient(
            current_r[:, 0], current_r[:, 1], current_r[:, 2], w_active
        )
        dr_ds = current_k / (be.unsqueeze_last(n) + 1e-12)
        dk_ds = grad_n
        return dr_ds, dk_ds

    for _ in range(max_steps):
        if not be.any(active_rays):
            break

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

        r[active_rays] = r_next_active
        k[active_rays] = k_next_active
        
        n_next, _ = grin_material.get_index_and_gradient(
            r_next_active[:, 0], r_next_active[:, 1], r_next_active[:, 2], w_active
        )
        opd[active_rays] = opd[active_rays] + 0.5 * (n_current + n_next) * step_size

        # --- Intersection Check ---
        segment_vec = r_next_active - r_active
        segment_len = be.linalg.norm(segment_vec, axis=-1)
        
        safe_segment_len = segment_len + 1e-12
        segment_dir = segment_vec / be.unsqueeze_last(safe_segment_len)

        # FIX: Instantiate RealRays using the correct constructor signature
        # (`intensity`, `wavelength`) as implied by the test suite and error messages.
        segment_rays = RealRays(
            x=r_active[:, 0], y=r_active[:, 1], z=r_active[:, 2],
            L=segment_dir[:, 0], M=segment_dir[:, 1], N=segment_dir[:, 2],
            intensity=be.ones_like(segment_len),
            wavelength=w_active
        )
        
        exit_surface.geometry.localize(segment_rays)
        distance_to_intersect = exit_surface.geometry.distance(segment_rays)
        
        intersected_mask_local = (distance_to_intersect > 1e-9) & (distance_to_intersect <= segment_len)

        if be.any(intersected_mask_local):
            active_indices_global = be.where(active_rays)[0]
            intersected_indices_global = active_indices_global[intersected_mask_local]

            intersection_point = r_active[intersected_mask_local] + \
                be.unsqueeze_last(distance_to_intersect[intersected_mask_local]) * segment_dir[intersected_mask_local]
            
            final_r[intersected_indices_global] = intersection_point
            final_k[intersected_indices_global] = k_next_active[intersected_mask_local]
            active_rays[intersected_indices_global] = False
    else:
        if be.any(active_rays):
            raise ValueError("Some rays did not intersect the exit surface after the maximum number of steps.")

    # --- Finalization ---
    rays_out = rays_in.copy()

    w_final = rays_in.w
    n_final, _ = grin_material.get_index_and_gradient(final_r[:, 0], final_r[:, 1], final_r[:, 2], w_final)
    final_d = final_k / (be.unsqueeze_last(n_final) + 1e-12)
    
    final_d_norm = be.linalg.norm(final_d, axis=-1)
    final_d /= (be.unsqueeze_last(final_d_norm) + 1e-12)

    rays_out.x, rays_out.y, rays_out.z = final_r[:, 0], final_r[:, 1], final_r[:, 2]
    rays_out.L, rays_out.M, rays_out.N = final_d[:, 0], final_d[:, 1], final_d[:, 2]
    rays_out.opd = opd
    rays_out.normalize()
    
    return rays_out