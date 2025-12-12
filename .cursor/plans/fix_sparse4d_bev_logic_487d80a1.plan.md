---
name: Fix Sparse4D BEV Logic
overview: Implement missing logic in `sparse4dbev` module including parameter updates, image size handling, instance bank time projection, and TopK sorting for anchor selection.
todos:
  - id: update-params
    content: Implement Sparse4DImpl::update_params
    status: pending
  - id: update-image-wh
    content: Implement image_wh update in forward
    status: pending
  - id: fix-dt-update
    content: Fix dt update in InstanceBank::project_anchors
    status: pending
  - id: implement-topk
    content: Implement TopK sorting in InstanceBank::update
    status: pending
---

# 1. Implement `update_params` in `sparse4d_impl.cpp`

- **Goal**: Enable dynamic updating of the `lidar2img` transformation matrix.
- **Details**:
    - Implement `Sparse4DImpl<T>::update_params`.
    - Accept `void*` param as `float*` (size: 6 * 4 * 4).
    - If `T` is `float`, directly copy to `lidar2img_` GPU buffer.
    - If `T` is `half`, convert `float` data to `half` (using `__float2half` or static cast wrapper) on CPU, then copy to `lidar2img_`.

# 2. Update `image_wh` in `sparse4d_impl.cpp`

- **Goal**: Provide correct image dimensions to the model.
- **Details**:
    - In `Sparse4DImpl<T>::forward`, before inference:
    - Retrieve image width and height from `config_` or `src_data`.
    - Create a host buffer containing `[W, H]` for each camera (6 cameras).
    - Convert to type `T` and upload to `image_wh_` GPU buffer.

# 3. Fix Time Interval in `InstanceBank::project_anchors`

- **Goal**: Ensure correct time interval (`dt`) is passed to the network.
- **Details**:
    - In `src/instance_bank/instance_bank.cpp`.
    - Convert `dt_` (float) to type `T` (handling `half` case).
    - Upload to `device_time_interval_`.

# 4. Implement TopK Sorting in `InstanceBank::update`

- **Goal**: Select the best anchors based on confidence scores.
- **Details**:
    - In `src/instance_bank/instance_bank.cpp`.
    - Replace the dummy indices loop.
    - Download `pred_confidence` (GPU) to Host.
    - Convert to `float` on Host (if `T` is `half`).
    - Perform `std::sort` or `std::partial_sort` to get top K indices.
    - Upload selected indices to `d_indices` (GPU).
    - Read `conf_decay` from `config_` instead of hardcoded `1.0f`.

# 5. Helper Utilities (if needed)

- **Goal**: Support FP16 conversions on host.
- **Details**:
    - Ensure `common/cuda_utils.hpp` or local helpers support `float` <-> `half` conversion if standard headers are insufficient on host.