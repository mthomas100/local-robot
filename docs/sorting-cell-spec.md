# The sorting cell: specification and acceptance tests

Written 2026-11-01, extended 2026-11-03 (Factory v2). This is the design the code in `factory/` implements: a virtual
factory cell that sorts real-world objects off a moving belt, built from models of real devices, and used as a vehicle
for every discipline a robotics engineer deals with, not only reinforcement learning. Each phase has an acceptance test
that was fixed before the phase's evaluation ran. Status as of 2026-11-03: F0-F7 accepted, F8-F10 not started.

## 1. The cell

| part | real device | simulation asset | tooling it implies on a real cell |
|---|---|---|---|
| arm | Universal Robots UR5e | MuJoCo Menagerie `universal_robots_ur5e` | ROS 2 `ur_robot_driver`, MoveIt 2, URSim, RTDE |
| gripper | Robotiq 2F-85, plus a suction cup on a slide for items wider than the stroke (hybrid tool) | Menagerie `robotiq_2f85`; vacuum = a weld that switches on only with a seal check | ROS 2 Robotiq driver |
| camera | Intel RealSense D435i, overhead, 1 m above the belt | Menagerie `realsense_d435i` geometry + a simulated RGB-D camera with its intrinsics, noise and latency | RealSense SDK conventions |
| belt | Dorner-class belt, 0.15 to 0.5 m/s, a 1 m reach zone | a treadmill: a plate on a slide joint, driven by a stiff velocity servo, teleported back every 0.4 m | none |
| objects | 11 items of the YCB Object and Model Set as stand-ins for waste: plastic (mustard, bleach, Windex bottles), metal (Master Chef, tomato soup, tuna, potted-meat cans), cardboard (cracker, sugar, pudding, gelatin boxes) | the YCB 16k-face textured scans, convex-decomposed with CoACD | the Amazon Picking Challenge / OCRTOC lineage |
| desk arm (F6) | SO-101 (six Feetech STS3215 hobby servos, LeRobot reference design) | Menagerie `robotstudio_so101` | LeRobot |

## 2. The job and the score

Mixed items ride the belt through the reach zone. Each must be picked and dropped into the chute for its material before
it leaves the zone; an item nobody picks falls off the end. **Score = items sorted into the right chute per minute**, at
a stated error budget. Every item ends in exactly one class: sorted, missed (never attempted), late (attempted, still
lost), wrong chute, dropped, knocked off. The belt never stops: that is what makes it a factory problem.

Difficulty ladder (each level is a scenario the generator can produce): L0 one item, belt stopped · L1 slow belt, one item
at a time · L2 random spacing, mixed materials · L3 clutter, items touching · L4 vision only, no privileged poses · L5 fast
belt, then a second arm.

## 3. The disciplines

Principle: **classical pipeline first, learning where it wins.** The first working cell uses no learning at all. A learned
module must beat the module it replaces on the score, on held-out objects, beyond a measured noise floor.

| # | discipline | what the cell builds | acceptance test |
|---|---|---|---|
| 1 | Cell design and layout | reach-envelope map (sampled forward kinematics), belt and chute placement inside it, cycle-time budget per belt speed | the reachable set with the zone and chutes inside; the budget table |
| 2 | Modelling and simulation | the MJCF cell; YCB meshes with mass, inertia, friction and convex collision; the treadmill; the vacuum model; solver checks | items ride the belt stably at every speed, no penetration, the same scene behaves alike in MuJoCo and Genesis |
| 3 | Kinematics | forward kinematics, damped-least-squares IK with joint limits, Jacobians, a singularity measure | IK reaches 1,000 random zone poses within 1 mm / 1° or reports them infeasible |
| 4 | Grasp planning | antipodal grasp candidates per scanned mesh, a ranked library, suction feasibility | every object has a grasp that holds under gravity and a shake test |
| 5 | Motion planning | Cartesian paths around chute walls and other items, time-parameterized under joint and tool limits | 0 collisions over 1,000 planned picks |
| 6 | Control | joint servo gains, gravity compensation, velocity feed-forward | tracking error and settling time measured, no oscillation |
| 7 | Perception | detection, identity and pose from the overhead depth camera; hand-eye calibration with planted error | error vs ground truth on held-out scenes; runs at the camera rate |
| 8 | State estimation and tracking | a Kalman filter per item, intercept prediction, latency compensation | intercept prediction error vs belt speed and latency |
| 9 | Scheduling | which item next under the deadline | throughput vs a greedy baseline on the same scenarios |
| 10 | Learning | (a) behaviour cloning from the classical pipeline; (b) RL for the transfer with shaping, curriculum and domain randomisation; (c) teacher-student distillation to a vision policy; (d) an agent-run reward-design loop | beats the classical baseline on L2+ with held-out objects, beyond the noise floor |
| 11 | Software architecture | modules with ROS-2-shaped interfaces, fixed loop rates, deterministic replay, configs in git | a recorded run replays bit for bit; a module swaps without touching the others |
| 12 | Safety and reliability | workspace and speed limits, collision checks, watchdogs, recovery (drop → retry), a failure taxonomy | every failure class detected, counted and recovered or safely stopped |
| 13 | Testing and evaluation | unit tests, scenario regression suites, KPIs, A/B tests with noise floors | the suite runs in minutes and gates every change |
| 14 | Sim-to-real | actuator and latency models, sensor noise, calibration error, system identification, ONNX at a fixed rate | the policy still works with the real-device models switched on |
| 15 | Operations | cycle time, throughput vs reliability, OEE accounting, an operator dashboard | a one-page report per cell version with the numbers and the trade-offs |

## 4. Phases and acceptance tests

| phase | builds | disciplines | done when | status |
|---|---|---|---|---|
| F0 the cell exists | UR5e + 2F-85 + belt + chutes + camera; YCB items on the belt in both engines | 1, 2 | items ride the belt in Genesis and MuJoCo; the reach map shows the zone inside the envelope | accepted 2026-11-02 |
| F1 classical pick, static | IK + grasp library + planner + joint control pick one item from the stopped belt into its chute | 3, 4, 5, 6, 11 | ≥ 95% on L0 over 200 trials across all items; the failure taxonomy exists | accepted 2026-11-02 |
| F2 classical pick, moving | tracking + intercept prediction + scheduling; the belt moves | 8, 9, 12 | throughput and miss rate measured at 0.15 and 0.3 m/s: the baseline to beat | accepted 2026-11-02 |
| F3 learned carry | an RL policy for the transfer (state-based), curriculum, domain randomisation, a hand-written warm start | 10, 13 | beats F2 on held-out objects beyond the noise floor in all four line conditions; a timeline video | accepted 2026-11-02 (attempt 3) |
| F4 seeing | classical perception from depth, tracking on estimates, a student carry without privileged state | 7, 8, 10 | L4 (vision only) within 10 points of L2 (privileged poses) | accepted 2026-11-02 |
| F5 the factory | clutter, vacuum for boxes, a dashboard, an operations report | 9, 11, 12, 15 | a 10-minute continuous run with the KPIs on the dashboard | accepted 2026-11-02 |
| F6 the desk arm, in sim | the SO-101 in a desk cell: reach map, payload-and-jaw table, 5-DoF IK, the classical pick; the **real-device kit** (hobby-servo loop, deadband, backlash, bus delay, torque limit, 12-bit encoders, camera latency, jitter, depth and RGB noise, a planted hand-eye error); **calibration**; **system identification** of planted plant parameters; a behaviour-cloned pick exported to **ONNX** and run at 30 Hz | 1-4, 6, 10a, 14 | with the kit on: classical ≥ 90% and ONNX ≥ 80% of 200 seeded L0 picks; system ID recovers every planted parameter within 10%; calibration recovers the planted error within 2 mm and 1°; the loop holds its rate | accepted 2026-11-03 (calibration, picks, rate: attempt 2; system ID: attempt 3) |
| F7 the fast belt | the operating envelope of the F5 cell (belt speed x load x clutter), then whatever makes 0.5 m/s work, each change measured alone | 6, 8, 9, 12, 15 | at 0.5 m/s v2 beats v1 beyond the noise floor (pre-registered, fresh seeds, 95% CI of the difference above 0) with 0 wrong chutes and no new protective-stop class, and is not worse at 0.3 m/s | accepted 2026-11-03 |
| F8 a second arm | a second UR5e on the same belt; a scheduler that splits items between the arms (greedy split first: the upstream arm skips what it cannot make, the downstream arm takes the rest); a shared-workspace guard tested with planted near-misses; per-arm KPIs | 1, 5, 9, 11, 12, 15 | on the F5 stream two arms sort ≥ 1.5x one arm's sorted/min on the same seeds (pre-registered, 95% CI above 0); a 10-minute two-arm shift with 0 arm-arm collisions and 0 wrong chutes | not started |
| F9 a learned grasp | the grasp learned on moving items, aimed at the failure classes the fast belt shows; its reward designed by a coding agent in a logged Eureka-style loop (4-8 candidates per round as code with named terms, fixed budgets, per-term statistics fed back, only strict improvements over a measured noise floor kept, survivors validated on held-out seeds); the success predicate, evaluation seeds and simulator hash-checked and never edited | 10b, 10d, 13 | beats the classical grasp on sorted/min at 0 wrong chutes on held-out items at 0.5 m/s beyond the noise floor; a ledger of at least two rounds; a timeline video | not started |
| F10 the v2 shift | everything on at once (two arms, the F7 changes, the F9 grasp where it won, vision only, the student carry, recovery, the real-device kit with UR5e parameters); deterministic replay; a scorecard of the fifteen disciplines' tests | 11, 13, 14, 15 | a 30-minute v2 shift beats the F5 cell's sorted/min beyond the noise floor with 0 wrong chutes and the kit on; a recorded run replays bit for bit; the scorecard and the v2 report are written | not started |

Rules that carry across phases: every acceptance run has a protocol committed before it runs (fresh seeds, the
comparison, the margin, a sample size from a power calculation when a pilot exists; see `factory/*/results/*-protocol*.md`);
the rendered video is part of the test, not only the metrics; every bug found becomes a regression test or a guard.

Not in scope: ROS 2 itself (the interfaces stay ROS-2-shaped) and the real SO-101 on a desk (optional, it needs the arm).
Open for anyone: the student carry's 3.5 / 6.3-point gap to the teacher (a pendulum model of the swing is the idea).
