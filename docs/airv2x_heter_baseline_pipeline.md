# AirV2X heterogeneous baselines (HEAL / STAMP / NegoCollab)

Camera-only. All LSS encoders are pretrained and **always frozen**.
Agent budget: `vehicle: 3`, `rsu: 2`, `drone: 2`.

## Shared Stage-0

Train each homogeneous collaborative detector **once**:

```
Vehicle base: 1–3 vehicles → frozen vehicle LSS → BEV backbone → PyramidFusion → heads
RSU base:     1–2 RSUs     → frozen RSU LSS     → BEV backbone → PyramidFusion → heads
Drone base:   1–2 drones   → frozen Drone LSS   → BEV backbone → PyramidFusion → heads
```

Commands:

```bash
python opencood/tools/train.py -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_vehicle_camera.yaml --tag homo_vehicle
python opencood/tools/train.py -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_rsu_camera.yaml --tag homo_rsu
python opencood/tools/train.py -y opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_drone_camera.yaml --tag homo_drone
```

## HEAL (AirV2X adaptation)

```
3 agent-specific frozen LSS
→ vehicle-domain shared BEV backbone
→ shared PyramidFusion / heads
```

Checkpoint mapping: encoders from matching bases; **shared detector from the vehicle base**.

```bash
python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_heal/airv2x_HEAL_collab_camera.yaml \
  --vehicle_dir <veh_base> --rsu_dir <rsu_base> --drone_dir <drone_base>
```

Trainable by default: shared backbone / fusion / heads (not LSS).

## STAMP (AirV2X adaptation)

```
3 agent-specific frozen LSS
→ frozen vehicle-domain shared backbone
→ agent-specific adapter
→ frozen shared fusion / head
```

```bash
python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_stamp/airv2x_stamp_collab_camera.yaml \
  --vehicle_dir <veh_base> --rsu_dir <rsu_base> --drone_dir <drone_base>
```

Trainable: adapters only. `stamp/single/*.yaml` are deprecated aliases of the homo bases.

## NegoCollab (method-local spaces)

```
3 independent local homogeneous bases
→ local backbone features
→ Sender → negotiated common space
→ ego Receiver
→ ego local PyramidFusion / head
```

Ego bypass: ego keeps the original local post-backbone feature.

```bash
# Stage 1 negotiation
python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage1_nego.yaml \
  --vehicle_dir <veh_base> --rsu_dir <rsu_base> --drone_dir <drone_base>

# Stage 2 receiver adaptation
python opencood/tools/train_airv2x_heter.py \
  -y opencood/hypes_yaml/airv2x/camera/det/airv2x_negocollab/airv2x_negocollab_stage2_ft.yaml \
  --vehicle_dir <veh_base> --rsu_dir <rsu_base> --drone_dir <drone_base> \
  --model_dir <stage1_log_dir>
```

Inference: same Stage-2 / `stage: inf` weights with `opencood/tools/inference_airv2x.py --model_dir <log_dir>`.

HEAL/STAMP follow the existing AirV2X shared-detector design. NegoCollab keeps three independent local perception stacks.
