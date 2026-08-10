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
    d_fwd=52.67,         # mm
    d_down= 178.375,    # mm
    shoulder_height=123.5,
    base_offset=33.5,
)

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

    def move_to_cartesian(self, x, y, z, slow = False):
        theta0, theta1, theta2, theta3 = inverse_kinematics(x, y, z, geom, elbow_up=False)
        print(f"Moving to Cartesian ({x}, {y}, {z}) -> joint angles (deg): "
              f"base: {166 + np.degrees(theta0):.1f}, "
              f"shoulder: {291 - np.degrees(theta1):.1f}, "
              f"elbow: {92.8 - np.degrees(theta2):.1f}, "
              f"wrist: {163 - np.degrees(theta3):.1f}")

        # {'base': 158.16849816849816, 'shoulder': 159.12087912087912, 'elbow': 174.06593406593407, 'wrist': 128.05860805860806, 'claw': 0.5128205128205128}
        speed = 500 if slow else 1000
        acc = 10 if slow else 40
        self.move_joint('base', 166 +  np.degrees(theta0), speed=speed, acc=acc)
        self.move_joint('shoulder', 291 - np.degrees(theta1), speed=speed, acc=acc)
        self.move_joint('elbow', 92.8 - np.degrees(theta2), speed=speed, acc=acc)
        self.move_joint('wrist', 163 - np.degrees(theta3), speed=int(speed * 1.8), acc=int(acc * 1.8))

    def stow(self):
        """Move the arm to a safe stowed position."""
        self.move_joint('base', 167.55)
        self.move_joint('shoulder', 116)
        self.move_joint('elbow', 265)
        self.move_joint('wrist', 0)
        self.move_joint('claw', 100)

    def move_to_square(self, square_name, z = 150, slow = False):
        """Move the arm to a specific chess square (e.g., 'e4')."""
        # Convert chess square to board coordinates (0-7 for x and y)
        col = ord(square_name[0]) - ord('a')  # 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
        row = 7 - (int(square_name[1]) - 1)     # '1' -> 0, '2' -> 1, ..., '8' -> 7

        # Map board coordinates to physical coordinates (in mm)
        # Assuming the bottom-left corner of the board is at (x=0, y=0)
        square_size_mm = 55  # adjust based on your actual board size
        arm_offset_mm = 90  # adjust if your arm's base is offset from the board's origin
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
        for z in [150, 125, 100, 75, 50, 20]:  # move down in steps
            self.move_to_square(square_name, z=z, slow=True)

        self.wait_for_move_completion()
        self.move_joint('claw', 98)  # open claw
        self.wait_for_move_completion()
        for z in [50, 75, 100, 150]:  # move down in steps
            self.move_to_square(square_name, z=z, slow=True)
        self.wait_for_move_completion()

    def pick(self, square_name, pawn = False):
        """Move to a square and close the claw to pick up a piece."""
        self.move_joint('claw', 98)  # open claw
        for z in [150, 125, 100, 75, 50, 15]:  # move down in steps
            self.move_to_square(square_name, z=z, slow=True)
        self.wait_for_move_completion()
        if pawn:
            self.move_joint('claw', 72)  # close claw for pawn
        else:
            self.move_joint('claw', 82)  # close claw for other pieces
        self.wait_for_move_completion()
        for z in [50, 75, 100, 150]:  # move down in steps
            self.move_to_square(square_name, z=z, slow=True)
        self.wait_for_move_completion()

    def from_to(self, from_square, to_square, pawn = False):
        """Pick up a piece from one square and drop it on another."""
        self.move_to_square(from_square)
        self.wait_for_move_completion()
        self.pick(from_square, pawn=pawn)

        self.move_to_square(to_square)
        self.wait_for_move_completion()
        self.drop(to_square)

if __name__ == "__main__":
    arm = ArmController()

    print("\nCurrent joint angles:")

    # arm.move_joint('claw', 69)
    # time.sleep(2)
    # arm.move_joint('claw', -5)

    # arm.move_joint('wrist', 150)
    # time.sleep(2)
    # arm.move_joint('wrist', 5)
   
    # arm.move_joint('elbow', 55)
    # time.sleep(5)
    # arm.move_joint('elbow', 90)

    # arm.move_joint('shoulder', 98)
    # time.sleep(15)
    # arm.move_joint('shoulder', 230)
    # arm.stow()
    # time.sleep(4)
    # arm.move_to_square('a8', above=False)
    # time.sleep(4)
    # arm.move_to_square('e4', above=True)
    # time.sleep(.5)
    # arm.move_to_square('h1', above=False)
    # time.sleep(6)
    # arm.move_to_square('e4', above=True)
    # time.sleep(.5)
    # arm.move_to_square('g8', above=False)
    # time.sleep(6)
    # arm.move_to_square('e4', above=True)
    # time.sleep(.5)
    # arm.move_to_square('e4', above=False)
    # time.sleep(6)
    # arm.stow()
    # time.sleep(3)
    # arm.stow()
    # time.sleep(3)
    # arm.move_to_square('h8', above=False)
    # time.sleep(7)
    # # arm.pick('e5', pawn=False)
    # arm.move_to_square('a8', above=False)
    # time.sleep(7)
    #     # arm.pick('e5', pawn=False)
    # arm.move_to_square('a1', above=False)
    # time.sleep(7)
    #     # arm.pick('e5', pawn=False)
    # arm.move_to_square('h1', above=False)
    # time.sleep(7)
    # # arm.drop('h6')
    # arm.stow()
    # arm.move_joint('base', 85)
    # time.sleep(2)
    # arm.move_joint('base', 194)
    # time.sleep(2)
    # arm.move_joint('base', 139.5)
    arm.from_to('e7', 'h4', pawn=False)
    arm.from_to('h4', 'e7', pawn=False)
    arm.stow()

    # for i in range(100):
    #     print(arm.read_all_angles())
    #     time.sleep(0.5)
    # print(arm.read_all_angles())

    # print("\nCentering all joints...")
    # arm.center_all()
    # time.sleep(2)

    # print("Moving base to 45 degrees...")
    # arm.move_joint('base', 125)
    # arm.move_joint('shoulder', 185)
    # arm.move_joint('elbow', 90)

    # time.sleep(5)
    # arm.move_joint('base', 125)
    # arm.move_joint('shoulder', 285)
    # arm.move_joint('elbow', 70)
    time.sleep(2)
    print(arm.read_all_angles())
    time.sleep(1)
    #arm.move_joint('shoulder', 175)
    time.sleep(2)
    # print("\nFinal joint angles:")
    print(arm.read_all_angles())
