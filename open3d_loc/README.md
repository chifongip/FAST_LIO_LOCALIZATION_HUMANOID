## Manual reset after map loss

If the estimated pose leaves the map, the local map crop can become empty.
The node skips registration for empty scan/map crops or missing target normals
and catches Open3D registration exceptions. It keeps processing odometry and
`/initialpose`, retains the current global correction, and reports zero
`/localization_3d_confidence` while registration is unavailable. A throttled
warning and an error status named `localization_registration` on
`/localization_3d_diagnostics` explain the failure.

Use RViz **2D Pose Estimate** with the fixed frame set to `map` to reset the
robot's position and heading. The reset forces a new map crop; registration
resumes when the crop and scan are usable. Odometry continues during map loss,
but the global pose remains based on the existing correction until reset or
successful registration.

The `test_registration_reset` regression uses a synthetic map and ROS inputs
to check startup outside the map, map loss after successful registration, and
recovery through `/initialpose`.

