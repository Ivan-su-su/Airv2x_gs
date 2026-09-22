# P1 camera cooperative training (train_multi_model)

Run these commands from the repository root on the training server. The three
`airv2x_homo_*` checkpoints already exist; no homo retraining is required.
Choose an actual `.pth` file for each agent. `P1_CKPT` must be the **same**
`net_final_branch_v45.pth` used for homo Stage-0 (the loader verifies every
frozen P1 tensor against all three homo checkpoints). Supply a separate
validation split; the old shared YAML pointed validation at training data.

```bash
export VEH_CKPT=/absolute/path/to/airv2x_homo_vehicle_camera/run/net_epochXX.pth
export RSU_CKPT=/absolute/path/to/airv2x_homo_rsu_camera/run/net_epochXX.pth
export DRONE_CKPT=/absolute/path/to/airv2x_homo_drone_camera/run/net_epochXX.pth
export P1_CKPT=/absolute/path/to/airv2x_gaussian_final/net_final_branch_v45.pth
export TRAIN_DATA=/absolute/path/to/airv2x/train
export VAL_DATA=/absolute/path/to/airv2x/validate

bash opencood/tools/train_heal.sh
bash opencood/tools/train_stamp.sh
bash opencood/tools/train_negocollab.sh stage1
```

After Stage 1 finishes, choose its `net_epoch_bestval_atXX.pth` (or a specific
`net_epochXX.pth`). Start a **new** Stage-2 run with fresh optimizer and schedule:

```bash
export NEGO_STAGE1_CKPT=/absolute/path/to/airv2x_negocollab_stage1_nego/run/net_epoch_bestval_atXX.pth
bash opencood/tools/train_negocollab.sh stage2
```

`--model_dir` is only for resuming the *same* stage from a checkpoint produced
by the updated trainer. It is not the Stage-1-to-Stage-2 handoff. Check the
load coverage reports and trainable parameter groups before committing to a
full run. HEAL trains the shared vehicle-domain detector, STAMP trains only
RSU/drone adapters, and NegoCollab trains communication/negotiation in Stage 1
then receiver modules in Stage 2. The STAMP implementation is an adapter
baseline, not the complete original STAMP protocol.
