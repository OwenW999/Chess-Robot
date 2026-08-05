"""
STS3215 servo control via Waveshare Bus Servo Adapter (A)
Run this directly on the machine the adapter is plugged into (Windows desktop).
"""

from st3215 import ST3215
import time

# --- CONFIG ---
PORT = 'COM3'  # <-- change this to match your Device Manager COM port


# bases forwards is 137.5* 

# Map joint names to servo IDs (adjust based on how you assigned IDs)
JOINT_IDS = {
    'base': 2,
    'shoulder': 3,
    # 'elbow': 4,
    # 'wrist': 5,
}

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

    def move_joint(self, joint_name, angle_deg, speed=550, acc=29):
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


if __name__ == "__main__":
    arm = ArmController()

    print("\nCurrent joint angles:")
    print(arm.read_all_angles())

    # print("\nCentering all joints...")
    # arm.center_all()
    # time.sleep(2)

    print("Moving base to 45 degrees...")
    arm.move_joint('base', 125)
    arm.move_joint('shoulder', 250)
    time.sleep(2)
    print(arm.read_all_angles())
    time.sleep(1)
    arm.move_joint('shoulder', 175)
    time.sleep(2)
    # print("\nFinal joint angles:")
    print(arm.read_all_angles())
