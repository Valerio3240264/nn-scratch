# Neural Network Classes Documentation

This document describes the **current implementation** under `classes/`.

## 1. Execution Graph Overview

The training graph is built from nodes implementing shared virtual interfaces:

`Input -> [Weights -> Activation] x N -> Loss`

Backward propagation starts at the loss node and recursively calls each predecessor node.

## 2. Core Interfaces (`classes/virtual_classes.h`)

- `BackwardClass`
  - Base interface for graph nodes that expose values, gradients, output shape, `backward()`, and `zero_grad()`.
- `WeightsClass`
  - Extends `BackwardClass` for affine modules with trainable parameters (`update()`), graph wiring (`set_pred`, `set_next`), and debug print helpers.
- `ActivationClass`
  - Extends `BackwardClass` for activation nodes with in-place forward via `operator()`.
- `LossClass`
  - Sink interface for scalar losses; computes loss and writes gradients to predecessor before propagating backward.

## 3. CPU Components

### `input` (`classes/cpu/headers/input.h`)

- Holds a non-owning `values` pointer and an owned gradient buffer.
- Shape metadata: `size` (features per sample), `batch_size`.
- `set_values()` rewires the input pointer.
- `backward()` is a leaf no-op.

### `weights` (`classes/cpu/headers/weights.h`)

- Stores:
  - `w`, `grad_w` with shape `(input_size, output_size)`
  - `b`, `grad_b` with shape `(output_size)`
- Forward:
  - Computes `Y = XW + b` for batched `X`.
- Backward:
  - `grad_w += X^T * dL/dY`
  - `dL/dX = dL/dY * W^T` written to predecessor gradient buffer
  - `grad_b += row_sum(dL/dY)`
- Initialization:
  - `TANH`/`SOFTMAX`/`LINEAR`: Xavier uniform `sqrt(6/(in+out))`
  - `RELU`: He uniform `sqrt(2/in)`

### `activation` (`classes/cpu/headers/activation.h`)

- Owned buffers: `value`, `grad`, both `(batch_size * size)`.
- Supported functions: `TANH`, `RELU`, `SOFTMAX`, `LINEAR`.
- Forward applies activation in-place on `value`; `SOFTMAX` uses stable row-wise normalization.
- Backward multiplies incoming gradient by local derivative and propagates to predecessor; `SOFTMAX` uses its row-wise Jacobian-vector product.

### `mse_loss` (`classes/cpu/headers/mse_loss.h`)

- Stores target buffer and scalar `loss_value`.
- Forward options:
  - dense target (`float*`)
  - class index targets (`size_t*`, internally one-hot encoded)
- Current loss formula:
  - `loss = sum((y - t)^2) / size`
- Current backward:
  - `dL/dy = (2/size) * (y - t)`

### `cross_entropy_loss` (`classes/cpu/headers/cross_entropy_loss.h`)

- Intended for classification with a `SOFTMAX` activation predecessor.
- Forward options:
  - dense one-hot/soft labels (`float*`)
  - class index targets (`size_t*`, internally one-hot encoded)
- Current loss formula:
  - `loss = -sum(t * log(y + 1e-15)) / batch_size`
- Current backward (w.r.t. softmax output):
  - `dL/dy = -(t / max(y, 1e-15)) / batch_size`

## 4. Layer and MLP Orchestration

### `layer` (`classes/mlp/headers/layer.h`)

- Wraps one weights node and one activation node.
- Supports CPU or CUDA backend (selected at construction).
- Forward: `W->operator()`, then `out->operator()`.
- `set_input()` rewires predecessor of the weights node.
- `zero_grad()` and `update()` delegate to weights.

### `mlp` (`classes/mlp/headers/mlp.h`)

- Owns:
  - dynamic array of `layer*`
  - one `loss_layer`
  - accumulated `current_loss`
- Supports per-layer activations and selectable loss (`MSE` or `CROSS_ENTROPY`).
- Two constructors:
  - full configuration (activations + loss + cuda flag)
  - legacy constructor (single activation for all layers, MSE)
- Typical training usage:
  1. `model(input_node)`
  2. `compute_loss(target)`
  3. `backward()`
  4. `update(lr)`
  5. `zero_grad()` / `zero_loss()` as needed
- `get_predictions()` requires the final activation to be `SOFTMAX`.

## 5. CUDA Components

### `cuda_input` (`classes/cuda/headers/cuda_input.cuh`)

- CUDA leaf input node.
- Non-owning `d_values` pointer set externally.
- Owned `d_grad` device buffer.
- `backward()` is a leaf no-op.

### `cuda_weights` (`classes/cuda/headers/cuda_weights.cuh`)

- Device affine module with `d_w`, `d_b`, `d_grad_w`, `d_grad_b`.
- Uses compact row-major matrices and bounds-checked tiled CUDA kernels.
- Forward and backward use CUDA matmul helpers.
- **Current constraint:** enforces `batch_size == 1`.

### `cuda_activation` (`classes/cuda/headers/cuda_activation.cuh`)

- Device activation with owned value/gradient buffers.
- Supports `TANH`, `RELU`, `SOFTMAX`, `LINEAR`.
- Backward launches activation derivative kernels then propagates.

### `cuda_mse_loss` (`classes/cuda/headers/cuda_mse_loss.cuh`)

- Device MSE loss with device-side target and reduction buffer.
- Supports dense targets and index targets (one-hot on device).
- Can own target memory or use external target pointer.
- **Current constraint:** enforces `batch_size == 1`.

### `cuda_cross_entropy_loss` (`classes/cuda/headers/cuda_cross_entropy_loss.cuh`)

- Device cross-entropy loss with device-side target and reduction buffer.
- Supports dense targets and index targets (one-hot on device).
- Can own target memory or use external target pointer.
- Intended for a `SOFTMAX` activation predecessor.
- **Current constraint:** enforces `batch_size == 1`.

## 6. Enums (`classes/enums.h`)

- `Activation_name`: `RELU`, `SIGMOID`, `TANH`, `SOFTMAX`, `LINEAR`
- `Loss_name`: `MSE`, `CROSS_ENTROPY`

> Note: `SIGMOID` exists in the enum but is not implemented by the generic CPU/CUDA `activation` classes.
