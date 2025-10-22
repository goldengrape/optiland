Of course. Here is the English translation of the development plan and to-do list.

---

### **Development Plan for GRIN Feature Integration and Architecture Refactoring**

**Goal:** To fully integrate GRIN functionality into the Optiland core tracing engine while migrating to a "Propagation Model" abstract architecture, ensuring the system is modular, extensible, and maintainable.

---

#### **Phase 1: Laying the Foundation – The Propagation Model Abstraction Layer**

The goal of this phase is to establish the new propagation model interface and refactor the existing straight-line propagation logic into its first concrete implementation. This paves the way for the subsequent GRIN integration with manageable impact on the existing system.

*   **[x] 1.1: Create the Propagation Model Abstract Base Class**
    *   **File:** `optiland/propagation/base.py`
    *   **Task:** Define the `PropagationModel` Abstract Base Class (ABC), containing one abstract method: `propagate(self, rays_in: RealRays, surface_in: BaseSurface, surface_out: BaseSurface) -> RealRays`.

*   **[x] 1.2: Implement the Homogeneous Medium Propagation Model**
    *   **File:** `optiland/propagation/homogeneous.py`
    *   **Task:**
        1.  Create the `HomogeneousPropagation(PropagationModel)` class.
        2.  Implement the `propagate` method. Its logic should replicate Optiland's current default behavior: calculate the geometric intersection of `rays_in` with `surface_out`, and update the optical path difference (`opd`) based on the refractive index `n` of `surface_in.material_post` and the geometric distance.
        3.  This logic may need to be extracted from the existing `Optic.trace` or `Surface.trace` methods.

---

#### **Phase 2: GRIN Integration – Encapsulating the Existing Algorithm**

The core of this phase is to encapsulate your completed and validated GRIN propagation code within the new propagation model class structure.

*   **[ ] 2.1: Migrate GRIN Module Files**
    *   **Task:** Ensure the following files are placed in their correct module paths and update the `__init__.py` files to expose the interfaces according to the final API design.
        *   `optiland/surfaces/gradient_surface.py` (containing `GradientBoundarySurface`)
        *   `optiland/materials/gradient_material.py` (containing `GradientMaterial`)

*   **[ ] 2.2: Create the GRIN Propagation Model**
    *   **File:** `optiland/propagation/gradient.py` (or rename and move `gradient_propagation.py` to this path)
    *   **Task:**
        1.  Create the `GrinPropagation(PropagationModel)` class.
        2.  Migrate the main body of your implemented `propagate_through_gradient` function into the `propagate` method of the `GrinPropagation` class.
        3.  **Adapt the interface:**
            *   The `propagate` method signature is `(self, rays_in, surface_in, surface_out)`.
            *   Inside the method, obtain the `GradientMaterial` instance via `grin_material = surface_in.material_post`.
            *   The `exit_surface` parameter is replaced by `surface_out`.
        4.  This class can accept `step_size` and `max_steps` as `__init__` parameters for easy configuration.

---

#### **Phase 3: Core Engine Refactoring – The "Heart Transplant"**

This is the most critical step, where we will modify Optiland's core ray tracing loop to recognize and dispatch different propagation models.

*   **[ ] 3.1: Locate and Analyze the Core Tracing Loop**
    *   **File:** `optiland/optic.py` (most likely) or a related `SurfaceGroup` class.
    *   **Task:** Find the main loop that iterates through the sequence of surfaces and calls their `trace()` methods.

*   **[ ] 3.2: Implement the Propagation Model Selector**
    *   **Task:** Inside the main loop, for the current surface `S_i` and the next surface `S_{i+1}`, implement a selection mechanism.
    *   **Logic:**
        1.  Get the medium between the two surfaces: `medium = S_i.material_post`.
        2.  Select the propagation model based on the `medium`'s type:
            ```python
            if isinstance(medium, GradientMaterial):
                propagation_model = GrinPropagation() # Or a pre-instantiated object
            else:
                propagation_model = HomogeneousPropagation()
            ```

*   **[ ] 3.3: Modify the Tracing Workflow**
    *   **Task:** Replace the original straight-line propagation logic with a call to the selected propagation model.
    *   **New Workflow:**
        1.  **Interaction:** Call `rays = S_i.trace(rays)`. This step handles the refraction/reflection of rays at the current surface. `GradientBoundarySurface` will inherit this method and correctly handle the boundary refraction using the `n0` value returned by `GradientMaterial.n()`.
        2.  **Propagation:**
            *   Get the next surface, `S_{i+1}`.
            *   Select the propagation model: `model = select_model(S_i.material_post)`.
            *   Call `rays = model.propagate(rays, S_i, S_{i+1})`.
        3.  The loop continues. `rays` are now located on the surface of `S_{i+1}`, ready for the next interaction.

---

#### **Phase 4: Validation and Testing**

Ensure the refactored system is not only functionally correct but also performant. The testing strategy from `GRIN_design_reference_context.md` is comprehensive and should be followed.

*   **[ ] 4.1: Unit Tests**
    *   **`GradientMaterial`:** Verify that the `get_index_and_gradient` method accurately computes values for known coordinates and coefficients.
    *   **`HomogeneousPropagation`:** Verify that its behavior is identical to the pre-refactor straight-line propagation results.
    *   **`GrinPropagation`:** Validate the accuracy of the RK4 algorithm's path and OPD accumulation against simple GRIN models with known analytical solutions (e.g., a linear gradient).

*   **[ ] 4.2: Integration Tests**
    *   Construct a complete `Optic` object containing a `GradientBoundarySurface` and `GradientMaterial` (e.g., a GRIN rod lens).
    *   Perform end-to-end ray tracing and check if the final ray coordinates, directions, and OPD match expectations.
    *   If possible, cross-validate the results with commercial software (Zemax, Code V).

*   **[ ] 4.3: Performance Benchmarking**
    *   Establish performance benchmarks by comparing the runtime of GRIN tracing versus standard tracing.
    *   Using Optiland's backend-switching feature, test the performance of GRIN tracing on both NumPy (CPU) and PyTorch (GPU) backends to validate the effectiveness of the vectorized implementation.

---

#### **Phase 5: Documentation and Examples**

Make the new feature easy for other developers and users to understand and use.

*   **[ ] 5.1: Update Code Documentation (Docstrings)**
    *   Write clear and complete docstrings for all newly created classes and methods (`PropagationModel`, `GrinPropagation`, etc.).

*   **[ ] 5.2: Create a Tutorial Example**
    *   **Task:** Write a new Jupyter Notebook tutorial, similar to `Tutorial_10a_Custom_Surface_Types.html`.
    *   **Content:** Demonstrate how to:
        1.  Instantiate a `GradientMaterial`.
        2.  Build a GRIN lens using `GradientBoundarySurface`.
        3.  Place this lens into an `Optic` system.
        4.  Perform a ray trace and visualize the results (e.g., a ray path diagram).

*   **[ ] 5.3: Update Project Documentation**
    *   Add a section to Optiland's official documentation explaining the GRIN feature and the new propagation model architecture.