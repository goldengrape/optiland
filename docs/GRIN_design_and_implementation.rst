.. _grin_design_and_implementation:

###################################################################
Optiland GRIN Functionality: Comprehensive Review and Implementation Guide
###################################################################

:Author: optiland fork
:Date: 2025-10-02
:Version: 3.0

.. contents:: Table of Contents
   :local:

*************************
1. Executive Summary
*************************

This report provides a final review of the design and implementation plan for introducing Gradient Refractive Index (GRIN) lens support into the Optiland project. We have comprehensively evaluated the original design document (Version 1.0) and an in-depth assessment report, confirming that this feature holds significant strategic importance for expanding Optiland's applications in cutting-edge fields such as biophotonics (especially human eye modeling).

The core strength of the original design lies in its strict adherence to the principles of **Axiomatic Design**, which decomposes a complex problem into three independent modules: Geometry (``Surface``), Physics (``Material``), and Behavior (``Propagation``). This decoupled design philosophy is a paradigm for building maintainable and extensible systems, which we fully endorse and will use as the cornerstone for all subsequent technical discussions.

Building upon this excellent architecture, this report integrates the key technical challenges and considerations from the in-depth assessment. Combined with best practices from **Design by Contract**, **Functional Programming**, and **Data-Oriented Programming**, we provide a more complete and precise final implementation blueprint.

*****************************************
2. Final Architecture and Module Definitions
*****************************************

We have adopted and refined the three core modules from the original design. The following are the final module definitions, incorporating Design by Contract and data-oriented principles to ensure code robustness, predictability, and elegance.

====================================================
2.1. DP1: ``GradientBoundarySurface`` (Geometric Domain)
====================================================

* **Responsibility**: This class acts as a simplified constructor for a standard surface (one with a `StandardGeometry`) that is intended to be used as a boundary for a GRIN medium. It serves as a "marker" for the ray tracing engine to identify the entrance to a GRIN medium.

* **Location**: ``optiland/surfaces/gradient_surface.py``

* **Final Code Definition**:

  .. code-block:: python

    """Defines a surface that marks the boundary of a gradient-index medium."""

    import optiland.backend as be
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries.standard import StandardGeometry
    from optiland.materials import IdealMaterial
    from optiland.surfaces.standard_surface import Surface


    class GradientBoundarySurface(Surface):
        """
        A surface that marks the entry into a Gradient Refractive Index (GRIN) medium.

        This class acts as a simplified constructor for a standard surface (one with
        a `StandardGeometry`) that is intended to be used as a boundary for a GRIN
        medium.

        Geometrically, this surface is identical to a standard spherical/conic surface.
        Its primary role is to be a distinct type that can trigger a special
        propagation model in the ray tracing engine. It does not contain any

        physical information about the gradient index itself.
        """

        def __init__(
            self,
            radius_of_curvature=be.inf,
            thickness=0.0,
            semi_diameter=None,
            conic=0.0,
            material_pre=None,
            material_post=None,
            **kwargs,
        ):
            """
            Initializes a GradientBoundarySurface.

            Args:
                radius_of_curvature (float, optional): The radius of curvature.
                    Defaults to infinity (a plane).
                thickness (float, optional): The thickness of the material following
                    the surface. Defaults to 0.0.
                semi_diameter (float, optional): The semi-diameter of the surface,
                    used for aperture clipping. Defaults to None.
                conic (float, optional): The conic constant. Defaults to 0.0.
                material_pre (BaseMaterial, optional): Material before the surface.
                    Defaults to ideal air (n=1.0).
                material_post (BaseMaterial, optional): Material after the surface.
                    Defaults to a default glass (n=1.5). This will typically be
                    replaced by a GradientMaterial by the tracing engine.
                **kwargs: Additional keyword arguments passed to the parent
                    `Surface` constructor.
            """
            cs = CoordinateSystem()  # Assumes a simple, non-decentered system
            geometry = StandardGeometry(cs, radius=radius_of_curvature, conic=conic)

            if material_pre is None:
                material_pre = IdealMaterial(n=1.0)
            if material_post is None:
                material_post = IdealMaterial(n=1.5)

            super().__init__(
                geometry=geometry,
                material_pre=material_pre,
                material_post=material_post,
                aperture=semi_diameter * 2 if semi_diameter is not None else None,
                **kwargs,
            )
            self.thickness = thickness

========================================================
2.2. DP2: ``GradientMaterial`` (Physical Property Domain)
========================================================

* **Responsibility**: Encapsulate the physical model of the GRIN medium, providing methods to calculate the refractive index and its gradient.

* **Location**: ``optiland/materials/gradient_material.py``

* **Final Code Definition**:

  .. code-block:: python

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

====================================================
2.3. DP3: ``GradientPropagation`` (Behavioral Domain)
====================================================

* **Responsibility**: Implement the ray propagation algorithm within the GRIN medium, the core of which is solving the differential equation for the ray's trajectory.

* **Location**: ``optiland/interactions/gradient_propagation.py``

* **Final Code Definition**:

  .. code-block:: python

    """
    Implements the ray propagation algorithm in a Gradient Refractive Index (GRIN) medium.
    It uses the RK4 numerical integration method to solve the ray equation: d/ds(n * dr/ds) = ∇n
    """
    import icontract
    import numpy as np
    from typing import Callable, Tuple

    # Assume Ray, BaseSurface, and GradientMaterial are defined elsewhere
    from optiland.rays import Ray
    from optiland.surfaces import BaseSurface
    from optiland.materials.gradient_material import GradientMaterial

    @icontract.require(lambda ray_in: ray_in.position.shape == (3,) and ray_in.direction.shape == (3,))
    @icontract.require(lambda step_size: step_size > 0)
    @icontract.require(lambda max_steps: max_steps > 0)
    @icontract.ensure(lambda result, exit_surface: exit_surface.contains(result.position, tol=1e-6), "Ray's endpoint must be on the exit surface")
    def propagate_through_gradient(
        ray_in: Ray,
        grin_material: "GradientMaterial",
        exit_surface: "BaseSurface",
        step_size: float = 0.1,
        max_steps: int = 10000
    ) -> Ray:
        """
        Traces a ray through a GRIN medium until it intersects the exit surface.

        Args:
            ray_in: The initial state of the ray (position and direction).
            grin_material: The physical model of the GRIN medium.
            exit_surface: The geometric surface marking the end of the GRIN medium.
            step_size: The step size for RK4 integration (in mm).
            max_steps: The maximum number of steps to prevent infinite loops.

        Returns:
            The final state of the ray at the exit surface.
        """
        r = ray_in.position.copy()
        n_start, _ = grin_material.get_index_and_gradient(r[0], r[1], r[2])
        k = n_start * ray_in.direction
        opd = 0.0

        def derivatives(current_r: np.ndarray, current_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            n, grad_n = grin_material.get_index_and_gradient(current_r[0], current_r[1], current_r[2])
            dr_ds = current_k / n if n != 0 else np.zeros(3)
            dk_ds = grad_n
            return dr_ds, dk_ds

        for i in range(max_steps):
            n_current = grin_material.get_index(r[0], r[1], r[2])
            
            # RK4 integration step
            r1, k1 = derivatives(r, k)
            r2, k2 = derivatives(r + 0.5 * step_size * r1, k + 0.5 * step_size * k1)
            r3, k3 = derivatives(r + 0.5 * step_size * r2, k + 0.5 * step_size * k2)
            r4, k4 = derivatives(r + step_size * r3, k + step_size * k3)

            r_next = r + (step_size / 6.0) * (r1 + 2*r2 + 2*r3 + r4)
            k_next = k + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Accumulate Optical Path Difference (OPD), estimated using the trapezoidal rule
            n_next = grin_material.get_index(r_next[0], r_next[1], r_next[2])
            opd += 0.5 * (n_current + n_next) * step_size
            
            # Check for intersection with the exit surface
            segment_vec = r_next - r
            segment_len = np.linalg.norm(segment_vec)
            if segment_len > 1e-9:
                segment_ray = Ray(position=r, direction=segment_vec / segment_len)
                distance_to_intersect = exit_surface.intersect(segment_ray)

                if 0 < distance_to_intersect <= segment_len:
                    intersection_point = r + distance_to_intersect * segment_ray.direction
                    n_final = grin_material.get_index(intersection_point[0], intersection_point[1], intersection_point[2])
                    final_direction = k_next / n_final
                    
                    # Final ray
                    ray_out = Ray(position=intersection_point, direction=final_direction / np.linalg.norm(final_direction))
                    ray_out.opd = ray_in.opd + opd # Assuming the Ray object has an opd attribute
                    return ray_out

            r, k = r_next, k_next

        raise ValueError("Ray did not intersect the exit surface after the maximum number of steps.")

***************************************************
3. Key Technical Considerations and Action Items
***************************************************

The assessment report accurately identified the core challenges from architectural design to engineering implementation. These issues must be explicitly addressed during development to ensure the correctness and efficiency of the GRIN functionality.

1.  **Integration Mechanism**:

      * **Question**: How does Optiland's core ray tracing engine (``Optic.trace``) identify and invoke ``propagate_through_gradient``?
      * **Recommendation**: In the ray tracing loop, check if the current surface is an instance of ``GradientBoundarySurface``. If so, its ``material_post`` property should be asserted to be an instance of ``GradientMaterial``. The tracing process must then determine the ``exit_surface`` and transfer control to ``propagate_through_gradient``.

2.  **GRIN Region Definition**:

      * **Question**: How is the scope of the GRIN medium defined? That is, how is the ``exit_surface`` determined?
      * **Option A (Recommended)**: Use paired markers. A GRIN region is defined by a ``GradientBoundarySurface`` (entry) and the next ``GradientBoundarySurface`` in the sequence (exit). This approach is clear and unambiguous.
      * **Option B**: Start from a ``GradientBoundarySurface`` and continue until the ``material_post`` of the next surface is no longer a ``GradientMaterial``. This option is more flexible but has a stronger dependency on the system sequence.
      * **Decision**: Option A is recommended for the initial implementation. This may require extending the ``Optic`` or ``SurfaceGroup`` class to manage these "surface pairs."

3.  **Boundary Refraction and Handover**:

      * **Question**: How is the ray's behavior handled at the moment it enters the GRIN medium?
      * **Recommendation**: The ``trace`` method of ``GradientBoundarySurface`` should be overridden. When a ray hits this surface, a standard Snell's Law refraction should be performed to calculate the ray's initial position and direction inside the medium. The refractive indices used for this calculation are that of ``material_pre`` and the index of the ``GradientMaterial`` at the intersection point (i.e., ``n0``). This new ray state is then passed as ``ray_in`` to the ``propagate_through_gradient`` function, ensuring a clear separation of responsibilities.

4.  **Algorithm Implementation Details**:

      * **Step Size Control**: The choice of step size for the RK4 algorithm is critical. A fixed step size is easy to implement but struggles to balance efficiency and accuracy.
          * **Short-term Plan**: Use a sufficiently small fixed ``step_size`` and expose it as a user-configurable parameter.
          * **Long-term Goal**: Implement an adaptive step size control algorithm (e.g., Runge-Kutta-Fehlberg, RKF45) that dynamically adjusts the step size based on local error, improving computational efficiency while guaranteeing precision.
      * **Optical Path Difference (OPD) Accumulation**: OPD is fundamental for wavefront analysis. As shown in the ``propagate_through_gradient`` code, ``∫n ds`` should be accumulated synchronously with each RK4 iteration.

5.  **Performance and Backend Integration**:

      * **Challenge**: GRIN tracing is far more computationally intensive than standard tracing.
      * **Recommendations**:
          * **Vectorization**: The ``get_index_and_gradient`` method in ``GradientMaterial`` must be designed from the outset to support NumPy vectorized operations, allowing it to process multiple rays simultaneously.
          * **GPU Acceleration**: Given Optiland's support for PyTorch, the core loop of ``propagate_through_gradient`` (especially the RK4 iteration and derivative calculations) should be implemented using PyTorch tensor operations. This not only leverages GPU acceleration but also paves the way for future automatic differentiation and optimization.
          * **JIT Compilation**: For maximum CPU performance, consider using Numba for Just-In-Time (JIT) compilation of computationally intensive functions.

6.  **Extensibility Considerations**:

      * **Dispersion**: The coefficients of the current ``GradientMaterial`` are constants. To support dispersion, these coefficients should be designed as functions or objects that accept a ``wavelength`` parameter, consistent with Optiland's existing material models. The ``get_index_and_gradient`` method will also need a ``wavelength`` parameter.
      * **Polynomial Form**: The current polynomial form is hard-coded. In the future, this could be abstracted into a configurable strategy, allowing users to define different gradient index models.

*************************
4. Conclusion and Outlook
*************************

The architectural design of this GRIN feature is excellent, fully embodying the principle of decoupling in software engineering. The implementation plan we have proposed, enhanced with Design by Contract and clear technical considerations, constitutes an actionable blueprint for development.

Successfully implementing this feature will equip Optiland with the ability to simulate complex biological optical systems (like the human eye) and design advanced optical components, greatly expanding its application scope and academic value. Future development should focus on resolving the specific "Key Technical Considerations," particularly regarding **integration with the core tracing logic**, **performance optimization of the RK4 algorithm (vectorization and GPU acceleration)**, and **support for dispersion**.

We firmly believe that by rigorously executing this thoroughly reviewed design plan, Optiland will take a significant step toward becoming a more powerful and professional top-tier open-source optical simulation tool.