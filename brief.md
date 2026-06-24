  # World-Action Model for Autonomous Driving Planning

  ## Project Goal

  Build a feasible, research-oriented autonomous driving project that demonstrates learned prediction/planning,
  world-action modeling, and optional VLM semantic conditioning.

  The project should help bridge background gaps for roles in:

  - autonomous driving ML
  - prediction and planning
  - world models / world-action models
  - embodied AI
  - robotics foundation models
  - VLM/VLA-adjacent applied research
  - general MLE roles requiring strong PyTorch, evaluation, and systems skills

  The project should be scoped for available resources:

  - Local workstation: 64 GB RAM, RTX 2080-class GPU
  - Remote compute: Clemson Palmetto HPC
  - Limited local training capacity
  - Cluster used for heavier training, preprocessing, sweeps, and evaluation

  ## Core Technical Claim

  This is a **world-action model project**, not just a trajectory prediction project.

  A normal prediction model learns:

  ```text
  past scene -> future scene

  This project learns:

  past scene + candidate ego action/trajectory -> future scene / risk / cost

  The model should answer:

  > If the ego vehicle takes this candidate action, how will the scene evolve and how risky is the outcome?

  The learned model is then used inside a planner to choose actions.

  ## Minimal System Loop

  driving scene
    -> generate candidate ego trajectories
    -> world-action model predicts outcome for each candidate
    -> planner scores predicted outcomes
    -> select safest / lowest-cost trajectory
    -> evaluate against baselines

  ## Recommended Dataset / Environment

  Start with logged-driving or lightweight planning datasets before attempting heavy simulation.

  Preferred starting options:

  1. Waymax / Waymo Open Motion-style data
      - Good for learned prediction/planning.
      - Useful for world-action modeling.
      - Less setup overhead than CARLA.

  2. nuPlan
      - Stronger planning benchmark credibility.
      - Heavier setup and data handling.

  3. CARLA
      - Useful later for demos.
      - Do not start here unless simulator setup is already solved.

  Initial recommendation:

  > Start with Waymax or a Waymo-style motion dataset abstraction. Add CARLA only after the core model/planner/
  > evaluation loop works.

  ## Local vs HPC Split

  ### Local Workstation

  Use local machine for:

  - repo development
  - config editing
  - small data samples
  - unit tests
  - smoke tests
  - one-batch model forward/backward checks
  - visualizations
  - plotting
  - report writing
  - README/blog/CV bullet drafting
  - debugging data schemas

  Local GPU should only run tiny experiments. Assume RTX 2080-class VRAM is limited.

  Local constraints:

  - avoid large batch sizes
  - avoid large VLM fine-tuning
  - avoid full benchmark sweeps
  - avoid long simulator rollouts

  ### Palmetto HPC

  Use Palmetto for:

  - dataset preprocessing
  - model training
  - larger batch experiments
  - hyperparameter sweeps
  - closed-loop or rollout-style evaluation
  - VLM embedding extraction
  - LoRA/post-training if added later
  - generating final experiment artifacts

  Cluster outputs should include:

  runs/
    experiment_name/
      config.yaml
      checkpoints/
      metrics.jsonl
      eval_summary.json
      plots/
      selected_rollouts/

  Only sync back selected checkpoints, metrics, plots, and videos.

  ## Proposed Repo Structure

  world-action-driving/
    README.md
    pyproject.toml
    configs/
      local_smoke.yaml
      train_wam.yaml
      eval_planner.yaml
    data/
      sample/
    src/
      datasets/
      models/
      planners/
      evaluators/
      viz/
      utils/
    scripts/
      prepare_data.py
      train.py
      evaluate.py
      visualize_rollout.py
    jobs/
      palmetto/
        train_wam.sbatch
        eval_planner.sbatch
    notebooks/
    reports/
    runs/

  runs/ and large data directories should be gitignored.

  ## Model Components

  ### 1. Scene Encoder

  Inputs:

  - ego history
  - nearby agent histories
  - lane/map features
  - traffic light state if available
  - optional rasterized BEV scene image

  Candidate architectures:

  - MLP baseline
  - Transformer over agents
  - graph/attention encoder
  - lightweight temporal encoder

  ### 2. Action Encoder

  Inputs:

  - candidate ego future trajectory
  - candidate control sequence
  - sampled lane-following trajectory
  - braking / acceleration / lane-change primitive

  Start simple:

  candidate ego trajectory: [x, y, yaw, speed] over H future steps

  ### 3. World-Action Model

  Core function:

  f(scene_history, candidate_ego_action) -> predicted_future, risk, cost

  Possible outputs:

  - future nearby-agent trajectories
  - occupancy grid
  - collision probability
  - off-road probability
  - progress score
  - comfort / jerk cost
  - scalar planning cost

  Start with scalar risk/cost + simple future trajectory prediction. Add richer outputs later.

  ### 4. Planner

  Start with sampling-based planning:

  1. Generate 64-256 candidate ego trajectories.
  2. Query world-action model for each candidate.
  3. Compute total cost.
  4. Choose best candidate.

  Candidate costs:

  total_cost =
    collision_weight * collision_risk
    + offroad_weight * offroad_risk
    + comfort_weight * jerk_cost
    - progress_weight * progress

  Later extensions:

  - MPC-style receding horizon
  - learned cost model
  - imitation-learned policy baseline
  - RL fine-tuning if feasible

  ## Baselines

  Implement baselines early.

  Required baselines:

  1. Constant velocity prediction
  2. Rule-based planner
  3. Imitation-only ego trajectory predictor
  4. World-action model + sampling planner

  Optional later:

  5. MPC with hand-coded dynamics
  6. RL or offline RL policy
  7. VLM-conditioned world-action model

  ## VLM Integration

  Do not make VLM the first milestone. Add it after the numeric planning pipeline works.

  Initial VLM role:

  BEV/image scene -> semantic tags/risk description -> conditioning feature

  Examples of semantic tags:

  - occluded pedestrian risk
  - unprotected left turn
  - construction zone
  - merging vehicle
  - cyclist nearby
  - crosswalk interaction
  - dense urban traffic
  - emergency braking scenario

  Model with VLM conditioning:

  f(scene_history, candidate_ego_action, semantic_context)
    -> future / risk / cost

  Possible VLM stages:

  1. Frozen VLM semantic annotator
  2. Cached VLM embeddings
  3. Auxiliary explanation head
  4. Lightweight LoRA/SFT on driving-scene QA data

  Avoid large VLM post-training until the main world-action model works.

  ## Milestones

  ### Milestone 0: Repo Initialization

  Deliverables:

  - project repo
  - environment file
  - README
  - configs
  - local smoke test
  - Palmetto job template

  Success condition:

  python scripts/train.py --config configs/local_smoke.yaml

  runs on a tiny sample without crashing.

  ### Milestone 1: Dataset + Visualization

  Deliverables:

  - dataset loader
  - small local sample
  - scene visualization
  - candidate trajectory visualization

  Success condition:

  - load one scene
  - render ego history, agents, lanes, and candidate ego trajectories

  ### Milestone 2: Baseline Planner

  Deliverables:

  - candidate trajectory sampler
  - rule-based cost function
  - constant-velocity prediction baseline

  Success condition:

  - choose a candidate trajectory for a scene
  - produce basic risk/progress metrics

  ### Milestone 3: First World-Action Model

  Deliverables:

  - action-conditioned model
  - supervised training loop
  - validation metrics
  - checkpoint saving

  Success condition:

  scene + candidate ego trajectory -> predicted risk/cost/future

  ### Milestone 4: Model-Based Planning

  Deliverables:

  - planner queries learned model over candidates
  - comparison against rule baseline
  - evaluation script

  Success condition:

  - report showing learned world-action planner vs baseline on held-out scenes

  ### Milestone 5: HPC Scaling

  Deliverables:

  - Palmetto sbatch scripts
  - reproducible config-based experiments
  - run artifact syncing instructions

  Success condition:

  - same training/evaluation runs on Palmetto
  - metrics copied back locally

  ### Milestone 6: VLM Semantic Conditioning

  Deliverables:

  - rendered BEV/image scene snapshots
  - frozen VLM semantic tag extraction
  - cached semantic features
  - ablation with and without VLM context

  Success condition:

  - show whether semantic conditioning improves risk/planning metrics or qualitative behavior

  ## Initial Codex Tasks

  When initializing the project, ask Codex to do the following:

  1. Create the repo structure.
  2. Add pyproject.toml with Python dependencies.
  3. Add config loading with YAML.
  4. Add a dummy dataset class and tiny synthetic sample.
  5. Add a candidate ego trajectory sampler.
  6. Add a simple rule-based planner.
  7. Add a placeholder world-action model in PyTorch.
  8. Add train.py, evaluate.py, and visualize_rollout.py.
  9. Add Palmetto sbatch templates.
  10. Add README with local and HPC workflows.

  ## Local Smoke Test Requirement

  The first implementation should run without any real dataset.

  Use synthetic scenes:

  - ego state history
  - 5-20 nearby agents
  - simple lane centerlines
  - generated candidate trajectories
  - synthetic collision/risk labels

  This avoids blocking on dataset setup.

  After the synthetic pipeline works, plug in real data.

  ## Palmetto Workflow

  Expected workflow:

  local:
    edit code
    run smoke tests
    commit changes

  sync to Palmetto:
    rsync repo
    prepare dataset
    submit sbatch training

  Palmetto:
    run training/eval
    write runs/

  local:
    sync metrics/plots/checkpoints back
    analyze and document

  ## What Not To Do Initially

  Avoid these at the beginning:

  - full CARLA setup
  - large VLM fine-tuning
  - large-scale RL
  - complex simulator integration
  - photorealistic world generation
  - overbuilding infrastructure before first model result

  The project should first prove:

  action-conditioned prediction improves planning evaluation

  Then add scale and VLM components.

  ## Resume Framing

  Potential resume bullet:

  > Built a world-action model for autonomous-driving planning, training an action-conditioned scene dynamics
  > model to evaluate candidate ego trajectories and integrating it with sampling/MPC-style planning; compared
  > against rule-based and imitation baselines using reproducible PyTorch training and HPC-scale evaluation.

  Optional VLM bullet:

  > Added VLM-based semantic scene conditioning for long-tail driving scenarios, using cached visual-language
  > annotations to improve risk-aware planning behavior under occlusion and interaction-heavy scenes.