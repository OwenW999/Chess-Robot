"""
STS3215 servo control via Waveshare Bus Servo Adapter (A)
Run this directly on the machine the adapter is plugged into (Windows desktop).
"""

from st3215 import ST3215
import time
from ik import inverse_kinematics, ArmGeometry, UnreachableTarget
import numpy as np
# --- CONFIG ---
PORT = 'COM3'  # <-- change this to match your Device Manager COM port


# bases forwards is 137.5* 

# TOTAL STOW: {'base': 165.89010989010987, 'shoulder': 116.4835164835165, 'elbow': 263.2087912087912, 'wrist': 0.7032967032967032, 'claw': 91.69230769230768}


# claw normal piece grab is 82 pawn is 75 and open is 98
# wrist full back is 5 forwards is 150   Range: (5, 150)
# elbow full back is 55 forwards is 219  Range: (55, 219)
# Shoulder full back is 98 forwards is 230  Range: (98, 230)
# base full right is 85 left is 194 Range: (85, 194)

# fully collpase 'shoulder': 172.45421245421244, 'elbow': 219
# Map joint names to servo IDs (adjust based on how you assigned IDs)
JOINT_IDS = {
    'base': 2,
    'shoulder': 3,
    'elbow': 4,
    'wrist': 5,
    'claw': 6,
}

geom = ArmGeometry(
    L1=232.5,          # mm
    L2=261.561,        # mm
    d_fwd=42, #52.67,         # mm
    d_down= 178.375,    # mm
    shoulder_height=123.5,
    base_offset=33.5,
)

# Height (mm) the claw travels at whenever it's moving between squares.
# Must clear the tallest piece on the board (usually the king) with margin.
# Measure your tallest piece and set this a bit above it -- 155 was the old
# default used for both "safe travel" and "hover before pick/drop", which is
# reused here so both concepts stay in sync instead of drifting apart.
TRAVEL_Z = 155

# --- Shoulder sag compensation ---
# The arm droops under gravity more the farther it reaches, so the shoulder
# needs to be commanded a bit higher (smaller angle) than IK says on long
# reaches to actually land where it's supposed to.
# TUNE THESE: reaches shorter than the threshold get zero compensation;
# beyond it, compensation grows linearly with reach. Increase SAG_GAIN if the
# claw is still landing low/short on far squares, decrease if it overshoots high.
SAG_REACH_THRESHOLD_MM = 250   # reach (mm) below which no compensation is applied
SAG_GAIN_DEG_PER_MM = 0.015     # extra degrees of "up" per mm of reach past the threshold

# Servo range: 0-4095 over the servo's full rotation (usually ~300 deg for STS3215)
SERVO_MAX_POS = 4095
SERVO_MAX_DEG = 360  # check your servo's actual rated range if unsure


class ArmController:
    def __init__(self, port=PORT):
        self.servo = ST3215(port)
        print("Scanning for servos...")
        found_ids = self.servo.ListServos()
        print(f"Found servo IDs: {found_ids}")

        # sanity check: make sure all expected joints are present
        missing = [name for name, sid in JOINT_IDS.items() if sid not in found_ids]
        if missing:
            print(f"WARNING: expected joints not found on bus: {missing}")

    def deg_to_pos(self, deg):
        """Convert degrees to raw servo position (0-4095)."""
        return int((deg / SERVO_MAX_DEG) * SERVO_MAX_POS)

    def pos_to_deg(self, pos):
        """Convert raw servo position back to degrees."""
        return (pos / SERVO_MAX_POS) * SERVO_MAX_DEG

    def move_joint(self, joint_name, angle_deg, speed=1000, acc=20):
        """Move a single joint to an angle in degrees."""
        if joint_name not in JOINT_IDS:
            raise ValueError(f"Unknown joint: {joint_name}")
        servo_id = JOINT_IDS[joint_name]
        pos = self.deg_to_pos(angle_deg)
        pos = max(0, min(SERVO_MAX_POS, pos))  # clamp to valid range
        self.servo.MoveTo(servo_id, pos, speed=speed, acc=acc)

    def move_all(self, base=None, shoulder=None, elbow=None, wrist=None):
        """Move multiple joints at once (only moves the ones you pass in)."""
        angles = {'base': base, 'shoulder': shoulder, 'elbow': elbow, 'wrist': wrist}
        for joint, angle in angles.items():
            if angle is not None:
                self.move_joint(joint, angle)

    def read_joint_angle(self, joint_name):
        """Read current angle of a joint in degrees."""
        servo_id = JOINT_IDS[joint_name]
        pos = self.servo.ReadPosition(servo_id)
        return self.pos_to_deg(pos)

    def read_all_angles(self):
        """Read all joint angles at once."""
        return {name: self.read_joint_angle(name) for name in JOINT_IDS}

    def center_all(self):
        """Send all servos to their center position (good default/home pose)."""
        for joint in JOINT_IDS:
            self.move_joint(joint, SERVO_MAX_DEG / 2, speed=100, acc=10)

    def shoulder_sag_compensation(self, x, y):
        """Extra degrees to subtract from the commanded shoulder angle (i.e. tilt
        further 'up') to counteract gravity sag, based on planar reach distance."""
        reach_mm = float(np.hypot(x, y))
        extra_reach = max(0.0, reach_mm - SAG_REACH_THRESHOLD_MM)
        return extra_reach * SAG_GAIN_DEG_PER_MM

    def move_to_cartesian(self, x, y, z, slow=False):
        theta0, theta1, theta2, theta3 = inverse_kinematics(x, y, z, geom, elbow_up=False)
        # print(f"Moving to Cartesian ({x}, {y}, {z}) -> joint angles (deg): "
        #       f"base: {166 + np.degrees(theta0):.1f}, "
        #       f"shoulder: {291 - np.degrees(theta1):.1f}, "
        #       f"elbow: {92.8 - np.degrees(theta2):.1f}, "
        #       f"wrist: {163 - np.degrees(theta3):.1f}")

        # {'base': 158.16849816849816, 'shoulder': 159.12087912087912, 'elbow': 174.06593406593407, 'wrist': 128.05860805860806, 'claw': 0.5128205128205128}
        speed = 500 if slow else 1000
        acc = 10 if slow else 40

        shoulder_angle = 293.0835 - np.degrees(theta1) - self.shoulder_sag_compensation(x, y)

        self.move_joint_timed('base', 168.5209 + np.degrees(theta0), duration=1)
        self.move_joint_timed('shoulder', shoulder_angle, duration=1)
        self.move_joint_timed('elbow', 90.9088 - np.degrees(theta2), duration=1)
        self.move_joint_timed('wrist', 162.5033 - np.degrees(theta3), duration=1)

    def stow(self):
        """
        Move the arm to a safe stowed position.

        Staged in two steps so the arm doesn't sweep sideways over the board
        at low height:
          1. Raise/fold shoulder + elbow first (lifts the claw up and back,
             base/wrist stay put) so the end effector clears all pieces.
          2. Only once lifted, rotate base and fold wrist/claw into their
             final stow angles.
        """
        lift_duration = 1.5
        fold_duration = 1.5

        # Step 1: lift the arm clear of the board before doing anything else.
        self.move_joint_timed('wrist', 0, 0.5)
        self.move_joint_timed('shoulder', 116, 1.3)
        self.move_joint_timed('elbow', 265, lift_duration)

        # Step 2: now that it's lifted, safe to swing base / fold wrist+claw.
        self.move_joint_timed('base', 167.55, fold_duration)
        self.move_joint_timed('claw', 100, fold_duration)
        self.wait_for_move_completion()

    def move_to_square(self, square_name, z=TRAVEL_Z, slow=False):
        """Move the arm to a specific chess square (e.g., 'e4')."""
        # Convert chess square to board coordinates (0-7 for x and y)
        col = ord(square_name[0]) - ord('a')  # 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
        row = 7 - (int(square_name[1]) - 1)     # '1' -> 0, '2' -> 1, ..., '8' -> 7

        # Map board coordinates to physical coordinates (in mm)
        # Assuming the bottom-left corner of the board is at (x=0, y=0)
        square_size_mm = 52  # adjust based on your actual board size
        arm_offset_mm = 85  # adjust if your arm's base is offset from the board's origin
        center_offset_mm = 211
        x = row * square_size_mm + square_size_mm / 2 + arm_offset_mm
        y = col * square_size_mm + square_size_mm / 2 - center_offset_mm

        self.move_to_cartesian(x, y, z, slow=slow)
        time.sleep(.11)

    def wait_for_move_completion(self, timeout=10.0, poll_interval=0.1):
        for name, sid in JOINT_IDS.items():
            start = time.time()
            while True:
                moving = self.servo.IsMoving(sid)
                if moving is None:
                    print(f"WARNING: comm error checking {name} (id {sid})")
                    break
                if not moving:
                    break
                if time.time() - start > timeout:
                    print(f"WARNING: {name} (id {sid}) timed out while moving")
                    break
                time.sleep(poll_interval)

    def drop(self, square_name):
        """Move to a square and open the claw to drop a piece."""

        self.move_to_square(square_name, z=20, slow=True)

        self.wait_for_move_completion()
        self.move_joint('claw', 92)  # open claw
        self.wait_for_move_completion()

        # Raise back to travel height (not an arbitrary hover height) so the
        # next move_to_square starts from a height that's known-clear of pieces.
        self.move_to_square(square_name, z=TRAVEL_Z, slow=True)
        self.wait_for_move_completion()

    def pick(self, square_name, pawn=False):
        """Move to a square and close the claw to pick up a piece."""
        self.move_joint('claw', 97)  # open claw

        self.move_to_square(square_name, z=20, slow=True)
        self.wait_for_move_completion()
        if pawn:
            self.move_joint('claw', 72)  # close claw for pawn
        else:
            self.move_joint('claw', 82)  # close claw for other pieces
        self.wait_for_move_completion()

        # Raise back to travel height (same reasoning as in drop()).
        self.move_to_square(square_name, z=TRAVEL_Z, slow=True)
        self.wait_for_move_completion()

    def from_to(self, from_square, to_square, pawn=False):
        """Pick up a piece from one square and drop it on another."""
        self.move_to_square(from_square)
        self.wait_for_move_completion()
        self.pick(from_square, pawn=pawn)

        self.move_to_square(to_square)
        self.wait_for_move_completion()
        self.drop(to_square)

    def move_joint_timed(self, joint_name, angle_deg, duration):
        servo_id = JOINT_IDS[joint_name]
        current_deg = self.read_joint_angle(joint_name)
        delta = abs(angle_deg - current_deg)
        delta_steps = self.deg_to_pos(delta)
        speed = max(25, int(delta_steps / duration)) if duration > 0 else 1000
        pos = max(0, min(SERVO_MAX_POS, self.deg_to_pos(angle_deg)))
        self.servo.MoveTo(servo_id, pos, speed=speed, acc=60)

    STS_P_COEF = 0x15   # 1 byte
    STS_D_COEF = 0x16   # 1 byte
    STS_I_COEF = 0x17   # 1 byte
    STS_MIN_START_FORCE = 0x18  # 2 bytes, little-endian
    STS_CW_DEADBAND = 0x1A      # 1 byte
    STS_CCW_DEADBAND = 0x1B     # 1 byte

    def tune_position_hold(self, sts_id, p=60, i=5, d=32, min_force=80):
        self.servo.writeTxRx(sts_id, self.STS_P_COEF, 1, [p])
        self.servo.writeTxRx(sts_id, self.STS_I_COEF, 1, [i])
        self.servo.writeTxRx(sts_id, self.STS_D_COEF, 1, [d])
        self.servo.writeTxRx(sts_id, self.STS_MIN_START_FORCE, 2,
                            [min_force & 0xFF, (min_force >> 8) & 0xFF])

    def set_deadband(self, sts_id, deadband=4):
        self.servo.writeTxRx(sts_id, self.STS_CW_DEADBAND, 1, [deadband])
        self.servo.writeTxRx(sts_id, self.STS_CCW_DEADBAND, 1, [deadband])

    def camera_clear(self):
        """Move the arm out of the camera's view so vision.get_board_grid() gets a clean read."""
        self.move_joint('base', 110.94505494505495)
        self.move_joint('shoulder', 116.74725274725276)
        self.move_joint('elbow', 240)
        self.move_joint('wrist', 0.7912087912087912)
        self.move_joint('claw', 92)


if __name__ == "__main__":
    arm = ArmController()

    # One-time tuning — comment out after running once
    # for name, sid in JOINT_IDS.items():
    #     if name in ('shoulder', 'elbow'):
    #         # these carry the most gravity load — tune more aggressively
    #         arm.tune_position_hold(sid, p=60, i=0, d=32, min_force=100)
    #         print(f"Tuned {name} (id {sid}) for position hold.")
    #     else:
    #         arm.tune_position_hold(sid, p=32, i=0, d=32, min_force=20)
    #     arm.set_deadband(sid, deadband=1)
    #     time.sleep(0.05)  # small gap between writes

    arm.from_to('e7', 'a1', pawn=True)
    arm.from_to('h8', 'b6', pawn=True)
    arm.from_to('b6', 'h2', pawn=True)
    arm.from_to('h2', 'e7', pawn=True)
    arm.from_to('a8', 'a4', pawn=True)
    arm.stow()

    for i in range(100):
        print(arm.read_all_angles())
        time.sleep(0.5)

    time.sleep(2)
    print(arm.read_all_angles())
    time.sleep(1)
    print(arm.read_all_angles())


#     """
# STS3215 servo control via Waveshare Bus Servo Adapter (A)
# Run this directly on the machine the adapter is plugged into (Windows desktop).
# """

# from st3215 import ST3215
# import time
# from ik import inverse_kinematics, ArmGeometry, UnreachableTarget
# import numpy as np
# # --- CONFIG ---
# PORT = 'COM3'  # <-- change this to match your Device Manager COM port


# # bases forwards is 137.5* 

# # TOTAL STOW: {'base': 165.89010989010987, 'shoulder': 116.4835164835165, 'elbow': 263.2087912087912, 'wrist': 0.7032967032967032, 'claw': 91.69230769230768}


# # claw normal piece grab is 82 pawn is 75 and open is 98
# # wrist full back is 5 forwards is 150   Range: (5, 150)
# # elbow full back is 55 forwards is 219  Range: (55, 219)
# # Shoulder full back is 98 forwards is 230  Range: (98, 230)
# # base full right is 85 left is 194 Range: (85, 194)

# # fully collpase 'shoulder': 172.45421245421244, 'elbow': 219
# # Map joint names to servo IDs (adjust based on how you assigned IDs)
# JOINT_IDS = {
#     'base': 2,
#     'shoulder': 3,
#     'elbow': 4,
#     'wrist': 5,
#     'claw': 6,
# }

# geom = ArmGeometry(
#     L1=232.5,          # mm
#     L2=261.561,        # mm
#     d_fwd=42, #52.67,         # mm
#     d_down= 178.375,    # mm
#     shoulder_height=123.5,
#     base_offset=33.5,
# )

# # Height (mm) the claw travels at whenever it's moving between squares.
# # Must clear the tallest piece on the board (usually the king) with margin.
# # Measure your tallest piece and set this a bit above it -- 155 was the old
# # default used for both "safe travel" and "hover before pick/drop", which is
# # reused here so both concepts stay in sync instead of drifting apart.
# TRAVEL_Z = 155

# # --- Shoulder sag compensation ---
# # The arm droops under gravity more the farther it reaches, so the shoulder
# # needs to be commanded a bit higher (smaller angle) than IK says on long
# # reaches to actually land where it's supposed to.
# # TUNE THESE: reaches shorter than the threshold get zero compensation;
# # beyond it, compensation grows linearly with reach. Increase SAG_GAIN if the
# # claw is still landing low/short on far squares, decrease if it overshoots high.
# SAG_REACH_THRESHOLD_MM = 250   # reach (mm) below which no compensation is applied
# SAG_GAIN_DEG_PER_MM = 0.02     # extra degrees of "up" per mm of reach past the threshold

# # Servo range: 0-4095 over the servo's full rotation (usually ~300 deg for STS3215)
# SERVO_MAX_POS = 4095
# SERVO_MAX_DEG = 360  # check your servo's actual rated range if unsure


# class ArmController:
#     def __init__(self, port=PORT):
#         self.servo = ST3215(port)
#         print("Scanning for servos...")
#         found_ids = self.servo.ListServos()
#         print(f"Found servo IDs: {found_ids}")

#         # sanity check: make sure all expected joints are present
#         missing = [name for name, sid in JOINT_IDS.items() if sid not in found_ids]
#         if missing:
#             print(f"WARNING: expected joints not found on bus: {missing}")

#         # Tracks the last (x, y, z) sent to move_to_cartesian, so stow() can
#         # lift straight up from wherever the arm currently is before folding.
#         self.last_xyz = None

#     def deg_to_pos(self, deg):
#         """Convert degrees to raw servo position (0-4095)."""
#         return int((deg / SERVO_MAX_DEG) * SERVO_MAX_POS)

#     def pos_to_deg(self, pos):
#         """Convert raw servo position back to degrees."""
#         return (pos / SERVO_MAX_POS) * SERVO_MAX_DEG

#     def move_joint(self, joint_name, angle_deg, speed=1000, acc=20):
#         """Move a single joint to an angle in degrees."""
#         if joint_name not in JOINT_IDS:
#             raise ValueError(f"Unknown joint: {joint_name}")
#         servo_id = JOINT_IDS[joint_name]
#         pos = self.deg_to_pos(angle_deg)
#         pos = max(0, min(SERVO_MAX_POS, pos))  # clamp to valid range
#         self.servo.MoveTo(servo_id, pos, speed=speed, acc=acc)

#     def move_all(self, base=None, shoulder=None, elbow=None, wrist=None):
#         """Move multiple joints at once (only moves the ones you pass in)."""
#         angles = {'base': base, 'shoulder': shoulder, 'elbow': elbow, 'wrist': wrist}
#         for joint, angle in angles.items():
#             if angle is not None:
#                 self.move_joint(joint, angle)

#     def read_joint_angle(self, joint_name):
#         """Read current angle of a joint in degrees."""
#         servo_id = JOINT_IDS[joint_name]
#         pos = self.servo.ReadPosition(servo_id)
#         return self.pos_to_deg(pos)

#     def read_all_angles(self):
#         """Read all joint angles at once."""
#         return {name: self.read_joint_angle(name) for name in JOINT_IDS}

#     def center_all(self):
#         """Send all servos to their center position (good default/home pose)."""
#         for joint in JOINT_IDS:
#             self.move_joint(joint, SERVO_MAX_DEG / 2, speed=100, acc=10)

#     def shoulder_sag_compensation(self, x, y):
#         """Extra degrees to subtract from the commanded shoulder angle (i.e. tilt
#         further 'up') to counteract gravity sag, based on planar reach distance."""
#         reach_mm = float(np.hypot(x, y))
#         extra_reach = max(0.0, reach_mm - SAG_REACH_THRESHOLD_MM)
#         return extra_reach * SAG_GAIN_DEG_PER_MM

#     def _solve_joint_angles(self, x, y, z):
#         """IK for a target (x, y, z) -> dict of target joint angles (deg),
#         including shoulder sag compensation. Shared by move_to_cartesian and
#         leave_folded_pose so both compute angles the same way."""
#         theta0, theta1, theta2, theta3 = inverse_kinematics(x, y, z, geom, elbow_up=False)
#         shoulder_angle = 293.0835 - np.degrees(theta1) - self.shoulder_sag_compensation(x, y)
#         return {
#             'base': 168.5209 + np.degrees(theta0),
#             'shoulder': shoulder_angle,
#             'elbow': 90.9088 - np.degrees(theta2),
#             'wrist': 162.5033 - np.degrees(theta3),
#         }

#     def move_to_cartesian(self, x, y, z, slow=False):
#         angles = self._solve_joint_angles(x, y, z)

#         self.move_joint_timed('base', angles['base'], duration=1)
#         self.move_joint_timed('shoulder', angles['shoulder'], duration=1)
#         self.move_joint_timed('elbow', angles['elbow'], duration=1)
#         self.move_joint_timed('wrist', angles['wrist'], duration=1)

#         self.last_xyz = (x, y, z)

#     def fold_in_stages(self, base, shoulder, elbow, wrist, claw, duration=1):
#         """
#         Move into a fully-folded pose (stow / camera_clear) in stages instead
#         of all at once. The tripod sits directly above the base, and these
#         folded poses hold the arm near-vertical right under it -- swinging a
#         still-extended arm straight up into that column is what was hitting
#         the tripod.

#         Order: curl elbow + wrist + claw in first (shortens the arm, base and
#         shoulder stay put) -> THEN raise the shoulder (now a short, compact
#         arm swings up near the tripod instead of an extended one) -> THEN
#         rotate the base last.
#         """
#         self.move_joint_timed('elbow', elbow, duration)
#         self.move_joint_timed('wrist', wrist, duration)
#         self.move_joint_timed('claw', claw, duration)
#         self.wait_for_move_completion()

#         self.move_joint_timed('shoulder', shoulder, duration)
#         self.wait_for_move_completion()

#         self.move_joint_timed('base', base, duration)
#         self.wait_for_move_completion()

#     def leave_folded_pose(self, x, y, z, duration=1):
#         """
#         Use this instead of move_to_cartesian/move_to_square for the FIRST
#         move after camera_clear() or stow(). It's the reverse of
#         fold_in_stages: rotate the base to the target orientation first
#         (arm still curled/raised near the tripod, but not extended), THEN
#         lower the shoulder, THEN extend the elbow/wrist out to reach the
#         target -- so the arm only extends once it's already away from the
#         tripod's column instead of swinging out through it.
#         """
#         angles = self._solve_joint_angles(x, y, z)

#         self.move_joint_timed('base', angles['base'], duration)
#         self.wait_for_move_completion()

#         self.move_joint_timed('shoulder', angles['shoulder'], duration)
#         self.wait_for_move_completion()

#         self.move_joint_timed('elbow', angles['elbow'], duration)
#         self.move_joint_timed('wrist', angles['wrist'], duration)
#         self.wait_for_move_completion()

#         self.last_xyz = (x, y, z)

#     def leave_folded_pose_square(self, square_name, z=TRAVEL_Z, duration=1):
#         """Same as leave_folded_pose but takes a chess square name."""
#         x, y = self._square_to_xy(square_name)
#         self.leave_folded_pose(x, y, z, duration=duration)

#     def stow(self):
#         """
#         Move the arm to a safe stowed position.

#           1. Lift straight up (Cartesian) at the current x/y so the claw
#              clears all pieces before anything else moves.
#           2. Fold into the stow angles in stages (see fold_in_stages) so the
#              arm stays compact while swinging up near the tripod.
#         """
#         # Step 1: lift straight up (in Cartesian space, at whatever x/y the
#         # arm is currently at) before touching the stow joint angles at all.
#         if self.last_xyz is not None:
#             x, y, _ = self.last_xyz
#             self.move_to_cartesian(x, y, TRAVEL_Z, slow=True)
#             self.wait_for_move_completion()
#         else:
#             print("WARNING: stow() called with no known position -- skipping pre-lift")

#         # Step 2: now that the claw is up above the pieces, fold into stow,
#         # staged to avoid clipping the tripod overhead.
#         self.fold_in_stages(base=167.55, shoulder=116, elbow=265, wrist=0, claw=100)

#     def _square_to_xy(self, square_name):
#         """Convert a chess square name (e.g. 'e4') to physical (x, y) mm."""
#         col = ord(square_name[0]) - ord('a')  # 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
#         row = 7 - (int(square_name[1]) - 1)     # '1' -> 0, '2' -> 1, ..., '8' -> 7

#         # Assuming the bottom-left corner of the board is at (x=0, y=0)
#         square_size_mm = 52  # adjust based on your actual board size
#         arm_offset_mm = 85  # adjust if your arm's base is offset from the board's origin
#         center_offset_mm = 211
#         x = row * square_size_mm + square_size_mm / 2 + arm_offset_mm
#         y = col * square_size_mm + square_size_mm / 2 - center_offset_mm
#         return x, y

#     def move_to_square(self, square_name, z=TRAVEL_Z, slow=False):
#         """Move the arm to a specific chess square (e.g., 'e4')."""
#         x, y = self._square_to_xy(square_name)
#         self.move_to_cartesian(x, y, z, slow=slow)
#         time.sleep(.11)

#     def wait_for_move_completion(self, timeout=10.0, poll_interval=0.1):
#         for name, sid in JOINT_IDS.items():
#             start = time.time()
#             while True:
#                 moving = self.servo.IsMoving(sid)
#                 if moving is None:
#                     print(f"WARNING: comm error checking {name} (id {sid})")
#                     break
#                 if not moving:
#                     break
#                 if time.time() - start > timeout:
#                     print(f"WARNING: {name} (id {sid}) timed out while moving")
#                     break
#                 time.sleep(poll_interval)

#     def drop(self, square_name):
#         """Move to a square and open the claw to drop a piece."""

#         self.move_to_square(square_name, z=15, slow=True)

#         self.wait_for_move_completion()
#         self.move_joint('claw', 92)  # open claw
#         self.wait_for_move_completion()

#         # Raise back to travel height (not an arbitrary hover height) so the
#         # next move_to_square starts from a height that's known-clear of pieces.
#         self.move_to_square(square_name, z=TRAVEL_Z, slow=True)
#         self.wait_for_move_completion()

#     def pick(self, square_name, pawn=False):
#         """Move to a square and close the claw to pick up a piece."""
#         self.move_joint('claw', 97)  # open claw

#         self.move_to_square(square_name, z=15, slow=True)
#         self.wait_for_move_completion()
#         if pawn:
#             self.move_joint('claw', 72)  # close claw for pawn
#         else:
#             self.move_joint('claw', 82)  # close claw for other pieces
#         self.wait_for_move_completion()

#         # Raise back to travel height (same reasoning as in drop()).
#         self.move_to_square(square_name, z=TRAVEL_Z, slow=True)
#         self.wait_for_move_completion()

#     def from_to(self, from_square, to_square, pawn=False):
#         """Pick up a piece from one square and drop it on another."""
#         self.move_to_square(from_square)
#         self.wait_for_move_completion()
#         self.pick(from_square, pawn=pawn)

#         self.move_to_square(to_square)
#         self.wait_for_move_completion()
#         self.drop(to_square)

#     def move_joint_timed(self, joint_name, angle_deg, duration):
#         servo_id = JOINT_IDS[joint_name]
#         current_deg = self.read_joint_angle(joint_name)
#         delta = abs(angle_deg - current_deg)
#         delta_steps = self.deg_to_pos(delta)
#         speed = max(25, int(delta_steps / duration)) if duration > 0 else 1000
#         pos = max(0, min(SERVO_MAX_POS, self.deg_to_pos(angle_deg)))
#         self.servo.MoveTo(servo_id, pos, speed=speed, acc=60)

#     STS_P_COEF = 0x15   # 1 byte
#     STS_D_COEF = 0x16   # 1 byte
#     STS_I_COEF = 0x17   # 1 byte
#     STS_MIN_START_FORCE = 0x18  # 2 bytes, little-endian
#     STS_CW_DEADBAND = 0x1A      # 1 byte
#     STS_CCW_DEADBAND = 0x1B     # 1 byte

#     def tune_position_hold(self, sts_id, p=60, i=5, d=32, min_force=80):
#         self.servo.writeTxRx(sts_id, self.STS_P_COEF, 1, [p])
#         self.servo.writeTxRx(sts_id, self.STS_I_COEF, 1, [i])
#         self.servo.writeTxRx(sts_id, self.STS_D_COEF, 1, [d])
#         self.servo.writeTxRx(sts_id, self.STS_MIN_START_FORCE, 2,
#                             [min_force & 0xFF, (min_force >> 8) & 0xFF])

#     def set_deadband(self, sts_id, deadband=4):
#         self.servo.writeTxRx(sts_id, self.STS_CW_DEADBAND, 1, [deadband])
#         self.servo.writeTxRx(sts_id, self.STS_CCW_DEADBAND, 1, [deadband])

#     def camera_clear(self):
#         """
#         Move the arm out of the camera's view so vision.get_board_grid() gets
#         a clean read. Staged (curl -> raise -> rotate) since the tripod sits
#         directly above the base -- see fold_in_stages for why.
#         """
#         self.fold_in_stages(
#             base=110.94505494505495,
#             shoulder=116.74725274725276,
#             elbow=240,
#             wrist=0.7912087912087912,
#             claw=92,
#         )


# if __name__ == "__main__":
#     arm = ArmController()

#     # One-time tuning — comment out after running once
#     # for name, sid in JOINT_IDS.items():
#     #     if name in ('shoulder', 'elbow'):
#     #         # these carry the most gravity load — tune more aggressively
#     #         arm.tune_position_hold(sid, p=60, i=0, d=32, min_force=100)
#     #         print(f"Tuned {name} (id {sid}) for position hold.")
#     #     else:
#     #         arm.tune_position_hold(sid, p=32, i=0, d=32, min_force=20)
#     #     arm.set_deadband(sid, deadband=1)
#     #     time.sleep(0.05)  # small gap between writes

#     arm.from_to('e7', 'h8', pawn=True)
#     arm.from_to('h8', 'b6', pawn=True)
#     arm.from_to('b6', 'h2', pawn=True)
#     arm.from_to('h2', 'e7', pawn=True)
#     arm.stow()

#     for i in range(100):
#         print(arm.read_all_angles())
#         time.sleep(0.5)

#     time.sleep(2)
#     print(arm.read_all_angles())
#     time.sleep(1)
#     print(arm.read_all_angles())