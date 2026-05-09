# Project 4: Progress-Conditioned ACT Rescue

This folder contains the code path for rescuing the blind SO101 space-bar imitation-learning task.

The source dataset stays immutable: `Carsamba/so101_blind_task2`. The derived dataset adds `observation.environment_state = [progress, 1 - progress, approach_vs_retract]`, so ACT can disambiguate approach, press, and retract phases without camera input.

## 1. Audit the Original Dataset

```powershell
python scripts/project4/audit_spacebar_dataset.py `
  --repo-id=Carsamba/so101_blind_task2
```

This writes:

- `outputs/project4_audit/spacebar_dataset_audit.json`
- `outputs/project4_audit/spacebar_state_action_trajectories.png`

## 2. Create the Progress Dataset

```powershell
python scripts/project4/augment_spacebar_progress_dataset.py `
  --output-repo-id=<hf_user>/so101_blind_task2_progress_v1 `
  --output-root=outputs/datasets/so101_blind_task2_progress_v1 `
  --overwrite
```

Add `--push-to-hub` after the local audit looks good.

## 3. Train ACT on Brev

```powershell
lerobot-train `
  --dataset.repo_id=<hf_user>/so101_blind_task2_progress_v1 `
  --policy.type=act `
  --policy.chunk_size=60 `
  --policy.n_action_steps=30 `
  --policy.use_vae=false `
  --batch_size=64 `
  --steps=10000 `
  --eval_freq=0 `
  --save_freq=1000 `
  --policy.device=cuda `
  --policy.push_to_hub=false `
  --wandb.enable=false
```

## 4. Offline Check the Checkpoint

```powershell
python scripts/project4/offline_act_progress_check.py `
  --policy-path=outputs/train/<run>/checkpoints/last/pretrained_model `
  --dataset-repo-id=<hf_user>/so101_blind_task2_progress_v1 `
  --episodes 0 5 10 19
```

Watch for a falling train loss, non-trivial predicted action standard deviation, and predicted trajectories that follow the recorded press/retract shape instead of collapsing to one middle pose.

## 5. Robot Rollout

Replay the original demo first:

```powershell
lerobot-replay `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --dataset.repo_id=Carsamba/so101_blind_task2 `
  --episode=0
```

Then run ACT with online progress injection:

```powershell
python scripts/project4/rollout_progress_act.py `
  --strategy.type=base `
  --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model `
  --robot.type=so101_follower `
  --robot.port=<robot_port> `
  --duration=10 `
  --fps=30 `
  --task="press the space bar"
```

The rollout script seeds the robot from episode 0 of the training dataset before starting policy control. If your checkpoint folder does not include `train_config.json`, add:

```powershell
--training_dataset_repo_id=<hf_user>/so101_blind_task2_progress_v1
```

## 6. Keyboard Hover Controller

The hover controller is the first non-contact bridge from Atakan's vision sandbox to the SO-101.
It does not press keys.

Inputs:

- accepted target-map JSON from `atakan-code`
- local Roboflow inference server for live wrist frames
- visible end effector in the wrist image
- SO-101 follower, wrist camera, and URDF path

Dry-run validation:

```powershell
python scripts/project4/keyboard_hover_controller.py `
  --dry-run `
  --target-map ..\atakan-code\outputs\target_map.json `
  --target-key h
```

Real hover run:

```powershell
python scripts/project4/keyboard_hover_controller.py `
  --target-map ..\atakan-code\outputs\target_map.json `
  --target-key h `
  --atakan-code-dir ..\atakan-code `
  --env-file ..\atakan-code\.env.inference `
  --robot-port COM_FOLLOWER `
  --robot-id so101_keyboard_follower `
  --camera-index 0 `
  --urdf-path path\to\so101_new_calib.urdf
```

The controller repeatedly detects the requested key and visible tip in the current wrist frame, estimates a local image-servo Jacobian from small lateral nudges, and stops once the tip-target pixel error is within `15 px` for three consecutive frames.

## 7. One-Command Handoff Setup

For Arda, run this from `arda-code` in PowerShell:

```powershell
.\scripts\project4\setup_keyboard_hover.ps1
```

Default inputs:

- sibling `..\atakan-code`
- Atakan branch `samuel`
- Arda branch `samuel-keyboard-hover`
- Python command `python`
- target key `h`
- generated fixture target map for dry-run validation

Default outputs:

- `..\atakan-code\outputs\setup_visual_servo_smoke.json`
- `outputs\setup_fixture_target_map.json`
- `outputs\setup_keyboard_hover_dry_run\hover_report.json`

Useful options:

```powershell
.\scripts\project4\setup_keyboard_hover.ps1 `
  -Python py `
  -TargetKey h `
  -StartInferenceServer `
  -RunLiveDetection
```

The default setup does not move the robot and does not require Roboflow to be running. It installs dependencies, runs Atakan unit tests, runs visual-servo simulation, checks the hover controller CLI, and validates an Arda dry-run target map.
