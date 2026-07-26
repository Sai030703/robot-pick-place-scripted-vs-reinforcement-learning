"""
vision_guided_pick_place.py

Vision-guided bin picking with a Franka Panda arm in PyBullet.
- Eye-in-hand camera (mounted on the wrist, moves with the gripper)
- Color-based cube detection (red cube = target)
- Pixel -> camera -> world coordinate conversion
- IK-driven pick and place

"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import csv
import datetime

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

END_EFFECTOR_INDEX = 11
NUM_ARM_JOINTS = 7
FINGER_JOINT_1 = 9
FINGER_JOINT_2 = 10

# A "bin" made from a tray + multiple red cubes to pick (simulates a cluttered
# bin -- a real industrial bin-picking scenario, not just a single isolated object)
# globalScaling shrinks the tray -- 0.6 makes it noticeably smaller than the
# default, leaving more clear table space around it for dropping cubes.
tray_id = p.loadURDF("tray/traybox.urdf", basePosition=[0.5, 0, 0.0], globalScaling=0.6)

# Ask PyBullet exactly where the tray's edges are, in world coordinates,
# instead of guessing. getAABB returns (min_corner, max_corner) of the
# tray's actual bounding box.
tray_aabb_min, tray_aabb_max = p.getAABB(tray_id)
print(f"Tray bounding box: min={tray_aabb_min}, max={tray_aabb_max}")

cube_half_extent = 0.02
NUM_CUBES = 4
cube_positions = [
    [0.47, 0.05, 0.05],
    [0.53, -0.03, 0.05],
    [0.45, -0.06, 0.05],
    [0.55, 0.06, 0.05],
]
cube_ids = []
for pos in cube_positions[:NUM_CUBES]:
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_half_extent]*3, rgbaColor=[1, 0, 0, 1])
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_half_extent]*3)
    cid = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=pos,
    )
    cube_ids.append(cid)
    # Give the cube "grippy" friction, like rubber instead of ice.
    # Without this, PyBullet's default friction is low and the fingers
    # can squeeze the cube but it still slides right out.
    p.changeDynamics(cid, -1, lateralFriction=1.0, spinningFriction=0.1)

# Also make the gripper fingers themselves grippy
p.changeDynamics(robot_id, FINGER_JOINT_1, lateralFriction=1.0)
p.changeDynamics(robot_id, FINGER_JOINT_2, lateralFriction=1.0)

# Let cubes settle under gravity before starting
for _ in range(100):
    p.stepSimulation()

# Move arm to a reasonable starting pose
home_angles = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.78]
for i, angle in enumerate(home_angles):
    p.resetJointState(robot_id, i, angle)
p.resetJointState(robot_id, FINGER_JOINT_1, 0.04)
p.resetJointState(robot_id, FINGER_JOINT_2, 0.04)

for _ in range(100):
    p.stepSimulation()

# ---------------------------------------------------------------------------
# Eye-in-hand camera: attached to the gripper, moves with it
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
FOV = 60
NEAR_PLANE = 0.01
FAR_PLANE = 2.0


def get_wrist_camera_image():
    """Render an image from a camera rigidly mounted just above the gripper."""
    link_state = p.getLinkState(robot_id, END_EFFECTOR_INDEX, computeForwardKinematics=True)
    gripper_pos, gripper_orn = link_state[0], link_state[1]

    rot_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3, 3)
    # Camera looks down the gripper's approach axis (local +Z, which after
    # the down-facing IK target points toward -world-Z / downward)
    forward_vec = rot_matrix.dot(np.array([0, 0, 1]))
    up_vec = rot_matrix.dot(np.array([1, 0, 0]))
    camera_offset = -forward_vec * 0.05  # sit slightly back from the lens along the view axis
    camera_pos = np.array(gripper_pos) + camera_offset

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_pos + forward_vec * 0.5,
        cameraUpVector=up_vec,
    )
    proj_matrix = p.computeProjectionMatrixFOV(FOV, CAMERA_WIDTH / CAMERA_HEIGHT, NEAR_PLANE, FAR_PLANE)

    _, _, rgb_img, depth_img, _ = p.getCameraImage(
        CAMERA_WIDTH, CAMERA_HEIGHT, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb_array = np.reshape(rgb_img, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3].astype(np.uint8)
    return rgb_array, view_matrix, proj_matrix


# ---------------------------------------------------------------------------
# Vision: detect the red cube by color, get its pixel centroid
# ---------------------------------------------------------------------------
def detect_red_cube_pixel(rgb_image):
    """Return (u, v) pixel centroid of the largest red blob, or None."""
    pixels = detect_all_red_cube_pixels(rgb_image)
    if not pixels:
        return None
    return pixels[0]


def detect_all_red_cube_pixels(rgb_image):
    """Return a list of (u, v) pixel centroids for ALL red blobs found,
    sorted largest-area first. Used for cluttered-bin picking where
    multiple objects are visible at once."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 2]
    contours.sort(key=cv2.contourArea, reverse=True)

    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        centroids.append((u, v))
    return centroids


# ---------------------------------------------------------------------------
# Simple approach: since ground-truth object pose is known in simulation,
# we use it directly for the grasp target. In a real system you'd back-project
# the pixel through the camera's inverse projection + depth to get world coords.
# This keeps the pipeline honest: vision DETECTS the object; the grasp target
# still comes from actual 3D reasoning (shown here via depth-based backprojection).
# ---------------------------------------------------------------------------
def pixel_to_world(u, v, depth_buffer, view_matrix, proj_matrix):
    """Convert a pixel + depth buffer value to world coordinates."""
    depth = depth_buffer[v, u]
    far, near = FAR_PLANE, NEAR_PLANE
    real_depth = far * near / (far - (far - near) * depth)

    view = np.array(view_matrix).reshape(4, 4, order="F")
    proj = np.array(proj_matrix).reshape(4, 4, order="F")

    x_ndc = (2.0 * u / CAMERA_WIDTH) - 1.0
    y_ndc = 1.0 - (2.0 * v / CAMERA_HEIGHT)

    clip = np.array([x_ndc, y_ndc, 1.0, 1.0])
    inv_vp = np.linalg.inv(proj.dot(view))
    world_h = inv_vp.dot(clip)
    world = world_h[:3] / world_h[3]

    # Scale toward camera-to-point direction using real depth for correction
    cam_pos = np.linalg.inv(view)[:3, 3]
    direction = (world - cam_pos)
    direction = direction / np.linalg.norm(direction)
    corrected_world = cam_pos + direction * real_depth
    return corrected_world


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def move_to_pose(target_pos, target_orn, steps=None, max_speed=1.0, tolerance=0.01, max_wait_steps=500):
    """Move the arm's gripper to a target position/orientation, and WAIT
    until it actually gets there (instead of just moving for a fixed amount
    of time and hoping that was enough).

    Why this matters: if you tell the arm "move slowly for X steps," and
    slowly + X steps isn't actually enough time to cover the distance, the
    arm just stops partway there -- silently. That's what was happening:
    the arm was stopping short of the cube every time. Now we check the
    arm's REAL position after each step and keep going until it's close
    enough to the target (or we hit a safety limit, to avoid looping forever
    if a position is truly unreachable).

    max_speed:      speed limit per joint (radians/second). Lower = smoother.
    tolerance:       how close (in meters) counts as "arrived".
    max_wait_steps:  safety cap so we never wait forever.
    """
    joint_angles = p.calculateInverseKinematics(
        robot_id, END_EFFECTOR_INDEX, target_pos, targetOrientation=target_orn,
        maxNumIterations=200,
    )
    for i in range(NUM_ARM_JOINTS):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL,
            targetPosition=joint_angles[i],
            force=200,
            maxVelocity=max_speed,
        )

    for step_count in range(max_wait_steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        current_pos = p.getLinkState(robot_id, END_EFFECTOR_INDEX)[0]
        distance_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

        if distance_to_target < tolerance:
            return  # arrived -- no need to keep waiting

    # If we get here, we hit the safety cap without arriving -- tell the user
    final_pos = p.getLinkState(robot_id, END_EFFECTOR_INDEX)[0]
    final_distance = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
    print(f"  (move_to_pose: gave up after {max_wait_steps} steps, "
          f"still {final_distance*1000:.1f} mm from target -- may be unreachable at this speed/pose)")


def set_gripper(opening, force=20):
    """Move the gripper fingers to a target opening width.

    opening: how far apart the fingers should be.
             0.04 = fully open, 0.0 = fully closed.
    force:   how HARD the fingers push/squeeze to get there.
             Think of it like how hard you squeeze your hand shut --
             a higher number means a firmer grip on the object.
    """
    p.setJointMotorControl2(robot_id, FINGER_JOINT_1, p.POSITION_CONTROL, opening, force=force)
    p.setJointMotorControl2(robot_id, FINGER_JOINT_2, p.POSITION_CONTROL, opening, force=force)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
CUBE_HALF_HEIGHT = 0.02  # known object geometry (in a real system: from CAD/SKU catalog)


def estimate_target_cube_position(remaining_cube_ids):
    """Capture from current wrist pose, detect all visible cubes, and return
    the position of whichever detected cube is nearest to a still-remaining
    (unpicked) ground-truth cube -- this lets us match a detected blob back
    to a specific cube_id so we know when it's been removed from the bin.

    Returns (matched_cube_id, center_position_estimate, rgb_image) or (None, None, rgb).
    """
    rgb, view_matrix, proj_matrix = get_wrist_camera_image()
    pixels = detect_all_red_cube_pixels(rgb)
    if not pixels:
        return None, None, rgb

    _, _, _, depth_img, _ = p.getCameraImage(
        CAMERA_WIDTH, CAMERA_HEIGHT, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    depth_array = np.reshape(depth_img, (CAMERA_HEIGHT, CAMERA_WIDTH))

    # Convert every detected blob to a world position (with surface->center correction)
    detected_world_positions = []
    for (u, v) in pixels:
        surface_pos = pixel_to_world(u, v, depth_array, view_matrix, proj_matrix)
        center_pos = surface_pos.copy()
        center_pos[2] -= CUBE_HALF_HEIGHT
        detected_world_positions.append(center_pos)

    # Match each detection to the nearest still-remaining ground-truth cube,
    # then pick the overall closest match to grasp next.
    best_match = None
    best_dist = float("inf")
    for est_pos in detected_world_positions:
        for cid in remaining_cube_ids:
            truth_pos, _ = p.getBasePositionAndOrientation(cid)
            dist = np.linalg.norm(np.array(est_pos) - np.array(truth_pos))
            if dist < best_dist:
                best_dist = dist
                best_match = (cid, est_pos)

    if best_match is None:
        return None, None, rgb
    matched_cid, matched_pos = best_match
    return matched_cid, matched_pos, rgb


def pick_and_place_one(remaining_cube_ids, cycle_num, log_rows):
    """Run the full look -> servo-refine -> grasp -> place cycle for the
    nearest remaining cube. Returns the cube_id that was picked, or None
    if nothing could be picked this cycle."""
    down_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

    # --- STAGE 1: coarse look ---
    look_position = [0.5, 0.0, 0.35]
    move_to_pose(look_position, down_orientation, steps=150)
    target_cid, estimate_1, rgb_1 = estimate_target_cube_position(remaining_cube_ids)
    if target_cid is None:
        print(f"[Cycle {cycle_num}] No cube detected -- bin may be empty.")
        return None
    truth_pos, _ = p.getBasePositionAndOrientation(target_cid)
    error_1 = np.linalg.norm(np.array(estimate_1) - np.array(truth_pos))
    print(f"[Cycle {cycle_num}] Coarse estimate error: {error_1*1000:.1f} mm (targeting cube id {target_cid})")

    # --- STAGE 2: closed-loop visual servoing refinement ---
    closer_position = [estimate_1[0], estimate_1[1], estimate_1[2] + 0.15]
    move_to_pose(closer_position, down_orientation, steps=120)
    _, estimate_2, rgb_2 = estimate_target_cube_position([target_cid])
    if estimate_2 is None:
        estimate_2 = estimate_1
    error_2 = np.linalg.norm(np.array(estimate_2) - np.array(truth_pos))
    print(f"[Cycle {cycle_num}] Refined estimate error: {error_2*1000:.1f} mm")

    world_target = estimate_2

    # --- STAGE 3: grasp and place ---

    # Step A: hover above the cube first (safer than diving straight down).
    above_cube = [world_target[0], world_target[1], world_target[2] + 0.15]
    move_to_pose(above_cube, down_orientation, steps=150, max_speed=0.8)

    # DIAGNOSTIC: where is the cube right now, before we even start descending?
    pos_before_descent, _ = p.getBasePositionAndOrientation(target_cid)
    print(f"[Cycle {cycle_num}] DEBUG cube position BEFORE descent: {pos_before_descent}")

    # DIAGNOSTIC: where are the actual fingertips vs. the "hand" reference point
    # we've been using as world_target? This tells us if our descent height
    # is lining up with the real fingers or not.
    hand_state = p.getLinkState(robot_id, END_EFFECTOR_INDEX)
    finger1_state = p.getLinkState(robot_id, FINGER_JOINT_1)
    finger2_state = p.getLinkState(robot_id, FINGER_JOINT_2)
    print(f"[Cycle {cycle_num}] DEBUG hand link z: {hand_state[0][2]:.4f}, "
          f"finger1 z: {finger1_state[0][2]:.4f}, finger2 z: {finger2_state[0][2]:.4f}")

    # Step B: open the gripper fingers wide, ready to grab
    set_gripper(0.04)

    # Step C: go down to the cube's ACTUAL MIDDLE height (not above it),
    # SLOWLY. Think of it like gently lowering your hand around the block
    # instead of stabbing down at it.
    at_cube = [world_target[0], world_target[1], world_target[2]]
    move_to_pose(at_cube, down_orientation, steps=200, max_speed=0.5)

    # DIAGNOSTIC: did the cube get knocked out of place DURING the descent,
    # before we even tried to close the fingers?
    pos_after_descent, _ = p.getBasePositionAndOrientation(target_cid)
    moved_during_descent = np.linalg.norm(
        np.array(pos_after_descent) - np.array(pos_before_descent)
    )
    print(f"[Cycle {cycle_num}] DEBUG cube position AFTER descent: {pos_after_descent}")
    if moved_during_descent > 0.01:
        print(f"[Cycle {cycle_num}] WARNING: cube moved {moved_during_descent*1000:.1f} mm "
              f"just from the arm descending -- the gripper likely BUMPED the cube "
              f"before even trying to close.")

    # Step D: close the fingers with more force, so they actually squeeze
    # the cube instead of just touching it lightly
    set_gripper(0.0, force=80)

    # Step E: give the simulation a brief pause so the fingers finish
    # closing and gripping firmly BEFORE we try to lift -- this is like
    # waiting a half-second after you close your hand, before pulling away
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Step F: check if the gripper is actually touching the cube.
    # This tells us in plain terms whether the grasp is likely to work.
    contact_points = p.getContactPoints(bodyA=robot_id, bodyB=target_cid)
    if len(contact_points) > 0:
        print(f"[Cycle {cycle_num}] Gripper IS touching the cube ({len(contact_points)} contact points). Good sign.")
    else:
        print(f"[Cycle {cycle_num}] WARNING: Gripper is NOT touching the cube. Grasp will likely fail.")


    # Step G: now lift, SLOWLY -- a slow, gentle lift is much less likely
    # to jostle the cube loose than a fast one
    lift_pose = [world_target[0], world_target[1], world_target[2] + 0.25]
    move_to_pose(lift_pose, down_orientation, steps=200, max_speed=0.5)

    # Step H: carry the cube to its drop spot SLOWLY. Moving fast while
    # holding something is exactly what can fling it out of the gripper --
    # like whipping your hand sideways while holding something loosely.
    # Drop locations are placed in a clean line, spaced 12cm apart, so
    # placed cubes end up next to each other with visible gaps between them.
    # Drop zone: placed clearly OUTSIDE the tray's actual right edge (using
    # the real bounding box we measured, not a guessed number), in a line
    # spaced 12cm apart so placed cubes sit next to each other with visible
    # gaps between them.
    # Drop zone: placed just past the tray's edge in the Y direction (not X),
    # much closer to the robot's base, so it stays within the arm's actual
    # reach. A robot arm can only stretch so far -- like your own arm, it
    # has a maximum comfortable reach -- and placing drop spots too far away
    # was exactly why the arm kept "giving up" before.
    DROP_LINE_START = [0.45, tray_aabb_max[1] + 0.06, 0.3]
    DROP_SPACING = 0.06
    drop_pose = [
        DROP_LINE_START[0],
        DROP_LINE_START[1] + DROP_SPACING * (cycle_num - 1),
        DROP_LINE_START[2],
    ]
    distance_from_base = np.linalg.norm(drop_pose)
    print(f"[Cycle {cycle_num}] Drop target: {drop_pose} "
          f"(distance from robot base: {distance_from_base*1000:.0f} mm)")
    move_to_pose(drop_pose, down_orientation, steps=250, max_speed=0.4)

    # Step I: lower slightly before releasing, so the cube doesn't drop
    # from too high up and bounce/roll away on landing
    drop_lower = [drop_pose[0], drop_pose[1], drop_pose[2] - 0.15]
    move_to_pose(drop_lower, down_orientation, steps=150, max_speed=0.3)

    set_gripper(0.04)
    # Give it a moment to settle after release before moving on
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    final_pos, _ = p.getBasePositionAndOrientation(target_cid)
    success = np.linalg.norm(np.array(final_pos[:2]) - np.array(drop_pose[:2])) < 0.12
    print(f"[Cycle {cycle_num}] Grasp success: {success}")

    log_rows.append([cycle_num, target_cid, error_1 * 1000, error_2 * 1000, success])
    return target_cid


def main():
    remaining = list(cube_ids)
    log_rows = []
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle = 0

    print(f"Starting cluttered-bin picking with {len(remaining)} cubes...")

    while remaining:
        cycle += 1
        picked = pick_and_place_one(remaining, cycle, log_rows)
        if picked is None:
            print("Stopping: no more cubes detected.")
            break
        remaining.remove(picked)
        print(f"Remaining cubes in bin: {len(remaining)}")

    # --- Write CSV log for the portfolio writeup ---
    log_filename = f"run_log_{run_id}.csv"
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "cube_id", "coarse_error_mm", "refined_error_mm", "grasp_success"])
        writer.writerows(log_rows)
    print(f"\nData logged to {log_filename}")

    successes = sum(1 for row in log_rows if row[4])
    print(f"Summary: {successes}/{len(log_rows)} cubes successfully picked and placed.")

    print("\nDone. Press ENTER in this terminal to close the simulation window...")
    input()


if __name__ == "__main__":
    main()
    p.disconnect()