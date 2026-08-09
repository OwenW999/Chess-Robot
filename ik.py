"""
Inverse kinematics for a 4-DOF arm (base yaw + 2-link planar arm + wrist)
with a claw offset that is NOT directly under the wrist joint, and a
claw that must always stay parallel to the ground.

Frame convention (world frame, matches the diagrams):
    x, y : horizontal plane
    z    : vertical, POSITIVE UP
    theta1 = 0 -> upper arm horizontal, POSITIVE = rotates upward
    theta2 = 0 -> forearm inline with upper arm (elbow straight),
              POSITIVE = forearm rotates upward relative to the upper arm
    theta3 = 0 -> offset link inline with forearm,
              solved so that (theta1 + theta2 + theta3) = 0, i.e. claw
              stays perfectly horizontal regardless of arm pose

    If your servo/encoder convention differs (e.g. z-down, or zero
    defined at a different pose), that's a constant offset added AFTER
    this function -- keep this math in the clean world frame and
    calibrate offsets separately per joint.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ArmGeometry:
    L1: float = 232.5       # upper arm length (shoulder -> elbow)
    L2: float = 261.561        # forearm length (elbow -> wrist)
    d_fwd: float = 52.67        # claw offset, horizontal, wrist -> claw tip
    d_down: float = 178.375    # claw offset, vertical drop, wrist -> claw tip
    shoulder_height: float = 123.5   # height of shoulder joint above base origin
    base_offset: float = 33.5        # horizontal offset from yaw axis to shoulder axis


class UnreachableTarget(Exception):
    pass


def inverse_kinematics(x: float, y: float, z: float,
                        geom: ArmGeometry,
                        elbow_up: bool = True):
    """
    Solve for joint angles that place the CLAW TIP at (x, y, z), world frame,
    with the claw kept parallel to the ground.

    Returns (theta0, theta1, theta2, theta3) in radians.
    Raises UnreachableTarget if the point is outside the arm's workspace.
    """

    # --- Step 1: base yaw is determined purely by target x, y ---
    theta0 = np.arctan2(y, x)

    # --- Step 2: back out the WRIST position from the claw target ---
    # The claw offset is fixed "straight ahead" in the claw's own frame,
    # and the claw never rotates out of the vertical plane defined by
    # theta0, so its world-frame projection only depends on theta0.
    offset_x = geom.d_fwd * np.cos(theta0)
    offset_y = geom.d_fwd * np.sin(theta0)
    offset_z = -geom.d_down

    wx = x - offset_x
    wy = y - offset_y
    wz = z - offset_z

    # --- Step 3: reduce to the 2-link planar problem ---
    r = np.hypot(wx, wy) - geom.base_offset
    zz = wz - geom.shoulder_height
    dist_sq = r * r + zz * zz
    dist = np.sqrt(dist_sq)

    max_reach = geom.L1 + geom.L2
    min_reach = abs(geom.L1 - geom.L2)
    if dist > max_reach or dist < min_reach:
        raise UnreachableTarget(
            f"distance to wrist target ({dist:.2f}) outside reachable "
            f"range [{min_reach:.2f}, {max_reach:.2f}]"
        )

    # --- Step 4: elbow angle (law of cosines) ---
    cos_theta2 = (dist_sq - geom.L1**2 - geom.L2**2) / (2 * geom.L1 * geom.L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # guard float noise at full reach
    theta2_mag = np.arccos(cos_theta2)
    theta2 = theta2_mag if elbow_up else -theta2_mag

    # --- Step 5: shoulder angle ---
    phi = np.arctan2(zz, r)
    alpha = np.arctan2(geom.L2 * np.sin(theta2), geom.L1 + geom.L2 * np.cos(theta2))
    theta1 = phi - alpha

    # --- Step 6: wrist angle keeps the claw level ---
    theta3 = -(theta1 + theta2)

    return theta0, theta1, theta2, theta3


def forward_kinematics_check(theta0, theta1, theta2, theta3, geom: ArmGeometry):
    """
    Sanity-check helper: recompute claw tip position from joint angles.
    Use this in a unit test to confirm inverse_kinematics() round-trips.
    """
    r = geom.L1 * np.cos(theta1) + geom.L2 * np.cos(theta1 + theta2)
    zz = geom.L1 * np.sin(theta1) + geom.L2 * np.sin(theta1 + theta2)

    # add wrist -> claw offset, using the fully-summed absolute angle
    claw_angle = theta1 + theta2 + theta3  # should be ~0
    wrist_r = r + geom.base_offset
    wrist_z = zz + geom.shoulder_height

    claw_r = wrist_r + geom.d_fwd * np.cos(claw_angle)
    claw_z = wrist_z + geom.d_fwd * np.sin(claw_angle) - geom.d_down * np.cos(claw_angle)

    x = claw_r * np.cos(theta0)
    y = claw_r * np.sin(theta0)
    z = claw_z

    return x, y, z


if __name__ == "__main__":
    # Example: tune these to your actual arm before trusting the output
    geom = ArmGeometry(
        L1=232.5,          # mm
        L2=261.561,        # mm
        d_fwd=0.0,         # mm
        d_down=178.375,    # mm
        shoulder_height=123.5,
        base_offset=33.5,
    )

    target = (300.0, 80.0, 20.0)  # x, y, z in mm, world frame

    for elbow in (True, False):
        try:
            t0, t1, t2, t3 = inverse_kinematics(*target, geom=geom, elbow_up=elbow)
            print(f"elbow_up={elbow}:")
            print(f"  theta0={np.degrees(t0):7.2f} deg")
            print(f"  theta1={np.degrees(t1):7.2f} deg")
            print(f"  theta2={np.degrees(t2):7.2f} deg")
            print(f"  theta3={np.degrees(t3):7.2f} deg")

            check = forward_kinematics_check(t0, t1, t2, t3, geom)
            print(f"  FK check -> {tuple(round(v, 2) for v in check)} "
                  f"(target was {target})")
        except UnreachableTarget as e:
            print(f"elbow_up={elbow}: unreachable -> {e}")