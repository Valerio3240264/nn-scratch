# Matrix Operations Utility (`MatricesOp`)

This file explains the math behind the matrix helper functions implemented in `utils/MatricesOp.cpp`.

All matrices are stored in **row-major** 1D arrays:

- Element `(i, j)` in a matrix with `M` columns is at `array[i * M + j]`.

---

## 1) Standard matrix multiplication

### `Multiply(A, B, R, N, K, M)`

Computes:

$
R = A B
$

with dimensions:

- $A \in \mathbb{R}^{N \times K}$
- $B \in \mathbb{R}^{K \times M}$
- $R \in \mathbb{R}^{N \times M}$

Element-wise:

$
R_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}
$

---

### `MultiplyAndAdd(A, B, C, R, N, K, M)`

Computes:

$
R = A B + C
$

with:

- $C \in \mathbb{R}^{N \times M}$

Element-wise:

$
R_{ij} = \left(\sum_{k=0}^{K-1} A_{ik} B_{kj}\right) + C_{ij}
$

---

### `InPlaceMultiplyAndAdd(A, B, R, N, K, M)`

Computes in-place:

$
R \leftarrow R + A B
$

Element-wise:

$
R_{ij} \leftarrow R_{ij} + \sum_{k=0}^{K-1} A_{ik} B_{kj}
$

`R` must already contain valid initial values.

---

## 2) Multiplication with transposed first matrix

These functions use $A^T$, useful when data is arranged as $A \in \mathbb{R}^{K \times N}$.

### `Multiply_Transpose1(A, B, R, N, K, M)`

Computes:

$
R = A^T B
$

with:

- $A \in \mathbb{R}^{K \times N}$ so $A^T \in \mathbb{R}^{N \times K}$
- $B \in \mathbb{R}^{K \times M}$
- $R \in \mathbb{R}^{N \times M}$

Element-wise:

$
R_{ij} = \sum_{k=0}^{K-1} A_{ki} B_{kj}
$

---

### `MultiplyAndAdd_Transpose1(A, B, C, R, N, K, M)`

Computes:

$
R = A^T B + C
$

Element-wise:

$
R_{ij} = \left(\sum_{k=0}^{K-1} A_{ki} B_{kj}\right) + C_{ij}
$

---

### `InPlaceMultiplyAndAdd_Transpose1(A, B, R, N, K, M)`

Computes in-place:

$
R \leftarrow R + A^T B
$

Element-wise:

$
R_{ij} \leftarrow R_{ij} + \sum_{k=0}^{K-1} A_{ki} B_{kj}
$

`R` must already contain valid initial values.

---

## 3) Multiplication with transposed second matrix

These functions use $B^T$, useful when data is arranged as $B \in \mathbb{R}^{M \times K}$.

### `Multiply_transpose2(A, B, R, N, K, M)`

Computes:

$
R = A B^T
$

with:

- $A \in \mathbb{R}^{N \times K}$
- $B \in \mathbb{R}^{M \times K}$ so $B^T \in \mathbb{R}^{K \times M}$
- $R \in \mathbb{R}^{N \times M}$

Element-wise:

$
R_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{jk}
$

---

### `MultiplyAndAdd_transpose2(A, B, C, R, N, K, M)`

Computes:

$
R = A B^T + C
$

Element-wise:

$
R_{ij} = \left(\sum_{k=0}^{K-1} A_{ik} B_{jk}\right) + C_{ij}
$

---

### `InPlaceMultiplyAndAdd_transpose2(A, B, R, N, K, M)`

Computes in-place:

$
R \leftarrow R + A B^T
$

Element-wise:

$
R_{ij} \leftarrow R_{ij} + \sum_{k=0}^{K-1} A_{ik} B_{jk}
$

`R` must already contain valid initial values.

---

## 4) Pointwise addition

### `Matrix_Add(A, B, R, N, M)`

Computes:

$
R = A + B
$

with $A, B, R \in \mathbb{R}^{N \times M}$, element-wise:

$R_{ij} = A_{ij} + B_{ij}$

---

### `InPlaceMatrix_Add(A, B, N, M)`

Computes in-place:

$A \leftarrow A + B$

Element-wise:

$A_{ij} \leftarrow A_{ij} + B_{ij}$

---

### `InPlaceVector_Add_Matrix(A, B, N, M)`

Computes in-place:

$A \leftarrow A + B$

Element-wise:

$A_{i} \leftarrow A_{i} + \sum_{j=0}^{M-1} B_{ij}$

$A \in \mathbb{R}^{N}$
$B \in \mathbb{R}^{N \times M}$

---

### `InPlaceVector_Add_MatrixT(A, B, N, M)`

Computes in-place:

$A \leftarrow A + B^T$

Element-wise:

$A_{i} \leftarrow A_{i} + \sum_{j=0}^{M-1} B_{ji}$

$A \in \mathbb{R}^{N}$
$B \in \mathbb{R}^{M \times N}$

---

## Why this utility exists

This utility centralizes low-level matrix math so weight class can focus on model logic (forward, backward, parameter updates) instead of repeating indexing and loop code.
