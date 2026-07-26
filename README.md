# Robotic Bin Picking: Scripted Control vs. Reinforcement Learning

Two approaches to the same core robotics problem — picking objects out of a bin with a Franka Panda arm in PyBullet — built to compare classical, hand-coded control against learned, reinforcement-learning-based control, including an honest investigation into where and why the learned approach breaks down under real camera-based perception.

## Why two approaches?

Most robotics portfolios show one working demo. This project deliberately builds two different solutions to the same problem so the tradeoffs between them are concrete and measured, not just described:

| | Scripted (Project 1) | Reinforcement Learning (Project 2) |
|---|---|---|
| Motion planning | Hand-coded IK waypoints | Learned via trial and reward (PPO) |
| Reliability (ground-truth state) | 4/4 cubes, sub-5mm targeting | 5/5 episodes, ~100mm lift height |
| Generalization | Only does exactly what's coded | Learns a general reaching/grasping strategy |
| Development effort | Explicit motion logic, manual tuning | Reward design, many training iterations |
| Real camera perception | Closed-loop visual servoing, sub-10mm accuracy | 0% success — see Stage 3 findings below |

---

## Project 1: Scripted Vision-Guided Bin Picking

A Franka Panda arm with a wrist-mounted (eye-in-hand) camera picks red cubes out of a cluttered bin using classical robotics techniques — no learning involved.

**Pipeline:** color-based cube detection (OpenCV/HSV) → depth back-projection to 3D coordinates → inverse kinematics → grasp → place in an organized line outside the bin.

**Result: 4/4 cubes successfully picked and placed, sub-5mm final targeting accuracy.**

### Debugging findings along the way
- **Camera orientation bug:** the wrist camera's "forward" axis was inverted, causing it to look away from the scene entirely — traced by logging actual gripper/camera position vs. expected, not by guessing.
- **Systematic depth bias:** depth sensors measure an object's *visible top surface*, not its center. This produced a consistent ~20mm targeting error (matching the cube's exact half-height) that did **not** improve with closer re-observation — the signature of a geometric bias, not sensor noise. Fixed with a known-object-height correction.
- **Motion timing bug:** capping arm speed without also waiting for actual arrival caused the arm to silently stop short of targets. Fixed by polling real position each step instead of assuming a fixed number of steps was "enough time."
- **Reachable workspace limits:** a drop zone placed outside the tray's measured bounding box turned out to exceed the arm's physical reach in one direction, causing repeated "unreachable target" failures — fixed by placing drop points based on actual measured distance from the robot's base.

---

## Project 2: Staged Reinforcement Learning

Instead of hand-coding motion, a PPO (Proximal Policy Optimization) policy is trained to control the arm through reward feedback alone. Built in three deliberate stages — each stage de-risking the next before adding complexity.

### Stage 1 — Reach
The AI learns only to move its gripper to a randomly placed 3D target — no grasping. Built first specifically to validate the full RL pipeline (environment design, reward, training loop) on an easier problem before tackling manipulation.

**Result: 5/5 success, ~48-50mm final accuracy** (clustered right at the success threshold — a direct result of the reward ending the episode the instant the threshold was crossed, with no incentive to do better than "just barely").

### Stage 2 — Pick and Lift (ground-truth object position)
Extended to grasp a cube and lift it 10cm. The AI's observations here include the cube's *exact* simulator-given position — this stage isolates whether the pick-and-lift task can be learned at all, before introducing real perception.

**Result: 5/5 success, consistent ~100-102mm lift heights, after fixing two separate reward-hacking bugs:**

1. **Touch-farming:** an early reward design gave +0.5 per step for merely touching the cube. The policy learned to sit there collecting that reward instead of attempting the harder, riskier lift. Fixed by making contact a one-time bonus instead of a repeating one.
2. **Hover-near-success:** even after fix #1, absolute-height-based reward meant an episode that hovered just under the success threshold for all 300 steps could out-earn an episode that succeeded and ended early (success terminates the episode, cutting off further reward). The policy learned to coast near — but never cross — the finish line. Fixed by rewarding lift **progress** (change in height) instead of height **held**.

Both bugs were identified from training/evaluation data (`ep_len_mean` staying at exactly 300 — meaning zero real successes during training — and reward totals that didn't match evaluation success/failure), not by assumption.

### Stage 3 — Pick and Lift (camera-based perception)
The same task, but the cube's position is no longer given directly by the simulator — it's estimated from the wrist camera using the same detection and depth back-projection pipeline as Project 1, including closed-loop refinement (look → move closer → look again).

**Result: 0/20 success. Average vision position error 30-150mm across evaluation runs**, compared to Project 1's sub-10mm accuracy under similar math.

**Root cause investigation:**
- The cube is 40mm wide; a vision error frequently *exceeding* the object's own size makes reliable grasping essentially impossible, regardless of how well the arm-control policy itself has learned.
- Training metrics (`explained_variance` ≈ 0.998-0.999, `ep_rew_mean` flat across millions of additional training steps) show the AI's internal value function converged almost perfectly — it successfully learned to predict outcomes. The ceiling on performance was set by the *inconsistency of the vision estimate itself* (accurate on some episodes, off by 200mm+ on others, with no learnable pattern distinguishing the two), not by insufficient training.
- Two depth-sampling fixes were tested empirically rather than assumed to work:
  - Averaging depth across the *entire* detected cube silhouette made results **worse** (101mm avg error, up from ~20-50mm) — traced to edge and side-face pixels (at a different depth than the flat top surface) contaminating the average.
  - Eroding the mask to sample only the object's center before averaging was tested next; given the ~1 hour cost per full retraining cycle and diminishing returns, iteration was stopped after this test rather than continued indefinitely.

**Conclusion:** this is a genuine, diagnosed limitation of **single-shot ("look-then-commit") perception**, not a training failure. The architecturally correct fix — informed directly by what already works in Project 1 — is **continuous closed-loop visual correction during the approach**, not a better one-time depth-sampling method. This was not implemented here due to time constraints; it's the clear next step.

---

## What this project demonstrates

- Building and debugging a full robotics pipeline: kinematics, computer vision, coordinate transforms, and closed-loop control
- Designing RL environments, observation/action spaces, and reward functions from scratch
- **Diagnosing subtle reward-hacking behavior from training data** (not guessing) and correcting it with principled reward redesign
- Understanding and demonstrating a fundamental RL/control principle: **task performance is bounded by observation quality**, no amount of additional training overcomes an information ceiling
- Honest engineering judgment about when further iteration has diminishing returns, and documenting a real limitation clearly instead of hiding it

## Limitations and future work

- All work is simulation-only (PyBullet); real-world deployment would face an additional sim-to-real gap (physics mismatch, sensor/actuator noise, latency) not addressed here
- Object detection uses color thresholding, not a learned vision model — sufficient for this project's scope, but not robust to varied object appearance
- Stage 3's single-shot perception limitation points directly to continuous visual servoing as the next concrete step
- Domain randomization (varying friction, mass, cube size during training) would likely improve robustness and is a natural extension

## Tech stack
PyBullet · Gymnasium · Stable-Baselines3 (PPO) · OpenCV · NumPy · Python 3.11


```

Each stage folder contains its environment definition, training script, evaluation script, and trained model checkpoint.
