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


# claw fully closed -5 open is 69
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
    d_fwd=0.0,         # mm
    d_down=178.375,    # mm
    shoulder_height=123.5,
    base_offset=33.5,
)

# Servo range: 0-4095 over the servo's full rotation (usually ~300 deg for STS3215)
SERVO_MAX_POS = 4095
SERVO_MAX_DEG = 300  # check your servo's actual rated range if unsure


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

    def move_joint(self, joint_name, angle_deg, speed=800, acc=50):
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

    def move_to_cartesian(self, x, y, z, elbow_up=True):
        theta0, theta1, theta2, theta3 = inverse_kinematics(x, y, z, geom, elbow_up=True)
        print(f"Moving to Cartesian ({x}, {y}, {z}) -> joint angles (deg): "
              f"base: {139.5 + np.degrees(theta0):.1f}, "
              f"shoulder: {273 - np.degrees(theta1):.1f}, "
              f"elbow: {46.8 - np.degrees(theta2):.1f}, "
              f"wrist: {167.9 - np.degrees(theta3):.1f}")
        
        # self.move_joint('base',139.5 +  np.degrees(theta0))
        # self.move_joint('shoulder', 273 - np.degrees(theta1))
        # self.move_joint('elbow', 46.8 - np.degrees(theta2))
        # self.move_joint('wrist', 167.9 - np.degrees(theta3))



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
    
    arm.move_to_cartesian(275, 115, 120, elbow_up=True)
    # arm.move_joint('base', 85)
    # time.sleep(2)
    # arm.move_joint('base', 194)
    # time.sleep(2)
    # arm.move_joint('base', 139.5)
     
    for i in range(100):
        print(arm.read_all_angles())
        time.sleep(0.5)
    print(arm.read_all_angles())

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
