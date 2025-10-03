# Pseudocode for Ray Tracing in GRIN Media using RK4

This document contains the pseudocode for an algorithm that traces a ray through a gradient refractive index (GRIN) medium using the 4th Order Runge-Kutta (RK4) method.

```
// -------------------------------------------------------------------------
// Function: F_derivative(z, v, params)
// Purpose: Calculate the state derivative vector F = dv/dz at a given point (z, v)
// Inputs:
//   z: current z-coordinate (float)
//   v: current state vector [x, y, Tx, Ty, OP] (5D array)
//   params: coefficients of the Gradient3 medium (struct or object)
// Output:
//   Derivative vector F (5D array), or an error flag (e.g., null) if total internal reflection occurs
// -------------------------------------------------------------------------
FUNCTION F_derivative(z, v, params):
    // 1. Unpack the state vector
    x = v[0], y = v[1], Tx = v[2], Ty = v[3]

    // 2. Efficiently calculate n and its partial derivatives
    r_sq = x*x + y*y
    r_sq_2 = r_sq * r_sq
    r_sq_3 = r_sq_2 * r_sq

    n = params.n0 + params.nr2*r_sq + params.nr4*r_sq_2 + params.nr6*r_sq_3 +
        params.nz1*z + params.nz2*z*z + params.nz3*z*z*z

    g_r = 2*params.nr2 + 4*params.nr4*r_sq + 6*params.nr6*r_sq_2
    dn_dx = x * g_r
    dn_dy = y * g_r

    // 3. Calculate the Hamiltonian H and check for validity
    n_sq = n*n
    T_sq_transverse = Tx*Tx + Ty*Ty

    radicand = n_sq - T_sq_transverse // Value inside the square root
    IF radicand <= 0 THEN
        // Total internal reflection or ray is perpendicular to the z-axis, cannot continue tracing with z as the step
        RETURN null // Return an error flag
    END IF

    H = -sqrt(radicand)

    // 4. Calculate and return the derivative vector F
    inv_H = 1.0 / H // Pre-calculate the inverse to avoid multiple divisions
    F = [
        -Tx * inv_H,
        -Ty * inv_H,
        -n * dn_dx * inv_H,
        -n * dn_dy * inv_H,
        -n_sq * inv_H
    ]
    RETURN F
END FUNCTION


// -------------------------------------------------------------------------
// Main procedure: RayTrace_Gradient3_RK4
// -------------------------------------------------------------------------
PROCEDURE RayTrace_Gradient3_RK4:
    // 1. Initialization
    // Define medium parameters
    gradient_params = {n0, nr2, nr4, nr6, nz1, nz2, nz3}

    // Define initial ray state
    z_start = 0.0
    z_end = 10.0
    step_size = 0.1

    initial_pos = [x0, y0]
    initial_dir = [dir_x0, dir_y0, dir_z0] // Unit direction vector

    // Calculate the initial state vector based on initial conditions
    n_start = calculate_n(z_start, initial_pos, gradient_params) // Requires a helper function
    Tx0 = n_start * initial_dir[0]
    Ty0 = n_start * initial_dir[1]

    z = z_start
    v = [initial_pos[0], initial_pos[1], Tx0, Ty0, 0.0] // [x, y, Tx, Ty, OP]

    // Store the trajectory
    ray_path = []
    ADD {z: z, state: v} TO ray_path

    // 2. RK4 main loop
    WHILE z < z_end:
        // Ensure the last step ends exactly at z_end
        current_step = min(step_size, z_end - z)

        // Calculate K1
        K1 = F_derivative(z, v, gradient_params)
        IF K1 IS null THEN BREAK // Error handling

        // Calculate K2
        v_temp2 = v + (current_step/2.0) * K1
        K2 = F_derivative(z + current_step/2.0, v_temp2, gradient_params)
        IF K2 IS null THEN BREAK // Error handling

        // Calculate K3
        v_temp3 = v + (current_step/2.0) * K2
        K3 = F_derivative(z + current_step/2.0, v_temp3, gradient_params)
        IF K3 IS null THEN BREAK // Error handling

        // Calculate K4
        v_temp4 = v + current_step * K3
        K4 = F_derivative(z + current_step, v_temp4, gradient_params)
        IF K4 IS null THEN BREAK // Error handling

        // 3. Update the state vector and z-coordinate
        v = v + (current_step / 6.0) * (K1 + 2*K2 + 2*K3 + K4)
        z = z + current_step

        ADD {z: z, state: v} TO ray_path

    END WHILE

    // 4. Output the results
    IF z >= z_end THEN
      PRINT "Ray tracing completed successfully."
      PRINT "Final state at z = ", z, ":"
      PRINT "  Position (x, y) = (", v[0], ", ", v[1], ")"
      PRINT "  Direction (Tx, Ty) = (", v[2], ", ", v[3], ")"
      PRINT "  Optical Path (OP) = ", v[4]
    ELSE
      PRINT "Ray tracing terminated early due to an error (e.g., total internal reflection) at z = ", z
    END IF

END PROCEDURE
```
