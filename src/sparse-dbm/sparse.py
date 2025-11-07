import mlx.core as mx
import numpy as np
import networkx as nx

masked_matmul_source = """
// Sparse matrix-matrix multiplication with mask
// This kernel computes C = mask * (A @ B) where A and B are in COO format
// Heavily optimized with SIMD and float32 accumulation
uint tid = thread_position_in_grid.x;
if (tid >= mask_nnz) return;

int i = mask_rows[tid];
int j = mask_cols[tid];

// Use float32 for accumulation to avoid precision loss
float sum = 0.0f;

// For each non-zero in A's row i - process 4 at a time
int a_idx = 0;
for (; a_idx + 3 < a_nnz; a_idx += 4) {
    // Load 4 row indices at once
    simd_int4 a_rows_vec = simd_int4(a_rows[a_idx], a_rows[a_idx+1], a_rows[a_idx+2], a_rows[a_idx+3]);
    simd_int4 i_vec = simd_int4(i);
    simd_float4 row_match = simd_float4(a_rows_vec == i_vec);

    for (int k = 0; k < 4; k++) {
        if (row_match[k] != 0.0f) {
            int a_k = a_cols[a_idx + k];
            float a_val = float(a_data[a_idx + k]);

            // Inner loop: find matching elements in B with SIMD
            int b_idx = 0;
            for (; b_idx + 3 < b_nnz; b_idx += 4) {
                simd_int4 b_rows_vec = simd_int4(b_rows[b_idx], b_rows[b_idx+1], b_rows[b_idx+2], b_rows[b_idx+3]);
                simd_int4 b_cols_vec = simd_int4(b_cols[b_idx], b_cols[b_idx+1], b_cols[b_idx+2], b_cols[b_idx+3]);
                simd_int4 ak_vec = simd_int4(a_k);
                simd_int4 j_vec = simd_int4(j);

                simd_float4 match = simd_float4((b_rows_vec == ak_vec) & (b_cols_vec == j_vec));
                simd_float4 b_vals = simd_float4(
                    float(b_data[b_idx]),
                    float(b_data[b_idx+1]),
                    float(b_data[b_idx+2]),
                    float(b_data[b_idx+3])
                );
                simd_float4 prod = match * b_vals;
                sum += a_val * (prod[0] + prod[1] + prod[2] + prod[3]);
            }

            // Handle remaining B elements
            for (; b_idx < b_nnz; b_idx++) {
                if (b_rows[b_idx] == a_k && b_cols[b_idx] == j) {
                    sum += a_val * float(b_data[b_idx]);
                }
            }
        }
    }
}

// Handle remaining A elements
for (; a_idx < a_nnz; a_idx++) {
    if (a_rows[a_idx] == i) {
        int k = a_cols[a_idx];
        float a_val = float(a_data[a_idx]);

        // Find matching column in B (k-th row of B) with SIMD
        int b_idx = 0;
        for (; b_idx + 3 < b_nnz; b_idx += 4) {
            simd_int4 b_rows_vec = simd_int4(b_rows[b_idx], b_rows[b_idx+1], b_rows[b_idx+2], b_rows[b_idx+3]);
            simd_int4 b_cols_vec = simd_int4(b_cols[b_idx], b_cols[b_idx+1], b_cols[b_idx+2], b_cols[b_idx+3]);
            simd_int4 k_vec = simd_int4(k);
            simd_int4 j_vec = simd_int4(j);

            simd_float4 match = simd_float4((b_rows_vec == k_vec) & (b_cols_vec == j_vec));
            simd_float4 b_vals = simd_float4(
                float(b_data[b_idx]),
                float(b_data[b_idx+1]),
                float(b_data[b_idx+2]),
                float(b_data[b_idx+3])
            );
            simd_float4 prod = match * b_vals;
            sum += a_val * (prod[0] + prod[1] + prod[2] + prod[3]);
        }

        for (; b_idx < b_nnz; b_idx++) {
            if (b_rows[b_idx] == k && b_cols[b_idx] == j) {
                sum += a_val * float(b_data[b_idx]);
            }
        }
    }
}

out_data[tid] = bfloat16_t(sum);
"""

sparse_vector_matmul_source = """
// Sparse matrix-vector multiplication: y = A @ x
// A is in COO format (rows, cols, data)
// x is a dense vector
// Heavily optimized with SIMD, float32 accumulation, and loop unrolling
uint tid = thread_position_in_grid.x;
if (tid >= n_rows) return;

// Use float32 for accumulation to reduce precision loss
float sum = 0.0f;
int tid_int = int(tid);

// Multiple SIMD accumulators for better instruction-level parallelism
simd_float4 acc0 = simd_float4(0);
simd_float4 acc1 = simd_float4(0);

// Sum all non-zeros in this row, process 8 at a time (2x unrolled 4-wide SIMD)
int i = 0;
for (; i + 7 < nnz; i += 8) {
    // First 4 elements
    simd_int4 row_vec0 = simd_int4(mat_rows[i], mat_rows[i+1], mat_rows[i+2], mat_rows[i+3]);
    simd_int4 tid_vec = simd_int4(tid_int);

    // Use select for branchless SIMD
    simd_float4 match0 = simd_float4(row_vec0 == tid_vec);
    simd_float4 vals0 = simd_float4(
        float(mat_data[i]) * float(vec[mat_cols[i]]),
        float(mat_data[i+1]) * float(vec[mat_cols[i+1]]),
        float(mat_data[i+2]) * float(vec[mat_cols[i+2]]),
        float(mat_data[i+3]) * float(vec[mat_cols[i+3]])
    );
    acc0 += match0 * vals0;

    // Second 4 elements
    simd_int4 row_vec1 = simd_int4(mat_rows[i+4], mat_rows[i+5], mat_rows[i+6], mat_rows[i+7]);
    simd_float4 match1 = simd_float4(row_vec1 == tid_vec);
    simd_float4 vals1 = simd_float4(
        float(mat_data[i+4]) * float(vec[mat_cols[i+4]]),
        float(mat_data[i+5]) * float(vec[mat_cols[i+5]]),
        float(mat_data[i+6]) * float(vec[mat_cols[i+6]]),
        float(mat_data[i+7]) * float(vec[mat_cols[i+7]])
    );
    acc1 += match1 * vals1;
}

// Reduce accumulators manually
sum = acc0[0] + acc0[1] + acc0[2] + acc0[3] + acc1[0] + acc1[1] + acc1[2] + acc1[3];

// Process remaining 4 elements with 4-wide SIMD
if (i + 3 < nnz) {
    simd_int4 row_vec = simd_int4(mat_rows[i], mat_rows[i+1], mat_rows[i+2], mat_rows[i+3]);
    simd_int4 tid_vec = simd_int4(tid_int);
    simd_float4 match = simd_float4(row_vec == tid_vec);
    simd_float4 vals = simd_float4(
        float(mat_data[i]) * float(vec[mat_cols[i]]),
        float(mat_data[i+1]) * float(vec[mat_cols[i+1]]),
        float(mat_data[i+2]) * float(vec[mat_cols[i+2]]),
        float(mat_data[i+3]) * float(vec[mat_cols[i+3]])
    );
    simd_float4 result = match * vals;
    sum += result[0] + result[1] + result[2] + result[3];
    i += 4;
}

// Handle remaining elements
for (; i < nnz; i++) {
    if (mat_rows[i] == tid_int) {
        sum += float(mat_data[i]) * float(vec[mat_cols[i]]);
    }
}

out[tid] = bfloat16_t(sum);
"""

masked_correlation_source = """
// Compute masked correlation: mask * (X.T @ X) / batch_size
// X is (batch_size, n_features), result is sparse where mask is non-zero
// Heavily optimized with SIMD, float32 accumulation, and loop unrolling
uint tid = thread_position_in_grid.x;
if (tid >= mask_nnz) return;

int i = mask_rows[tid];
int j = mask_cols[tid];

// Multiple SIMD accumulators for better instruction-level parallelism
simd_float4 acc0 = simd_float4(0);
simd_float4 acc1 = simd_float4(0);
simd_float4 acc2 = simd_float4(0);
simd_float4 acc3 = simd_float4(0);
float sum = 0.0f;

// Compute dot product with aggressive unrolling - 16 elements at a time
int b = 0;
for (; b + 15 < batch_size; b += 16) {
    // Unroll 4x for better ILP
    simd_float4 xi0 = simd_float4(
        float(X[(b + 0) * n_features + i]),
        float(X[(b + 1) * n_features + i]),
        float(X[(b + 2) * n_features + i]),
        float(X[(b + 3) * n_features + i])
    );
    simd_float4 xj0 = simd_float4(
        float(X[(b + 0) * n_features + j]),
        float(X[(b + 1) * n_features + j]),
        float(X[(b + 2) * n_features + j]),
        float(X[(b + 3) * n_features + j])
    );
    acc0 += xi0 * xj0;

    simd_float4 xi1 = simd_float4(
        float(X[(b + 4) * n_features + i]),
        float(X[(b + 5) * n_features + i]),
        float(X[(b + 6) * n_features + i]),
        float(X[(b + 7) * n_features + i])
    );
    simd_float4 xj1 = simd_float4(
        float(X[(b + 4) * n_features + j]),
        float(X[(b + 5) * n_features + j]),
        float(X[(b + 6) * n_features + j]),
        float(X[(b + 7) * n_features + j])
    );
    acc1 += xi1 * xj1;

    simd_float4 xi2 = simd_float4(
        float(X[(b + 8) * n_features + i]),
        float(X[(b + 9) * n_features + i]),
        float(X[(b + 10) * n_features + i]),
        float(X[(b + 11) * n_features + i])
    );
    simd_float4 xj2 = simd_float4(
        float(X[(b + 8) * n_features + j]),
        float(X[(b + 9) * n_features + j]),
        float(X[(b + 10) * n_features + j]),
        float(X[(b + 11) * n_features + j])
    );
    acc2 += xi2 * xj2;

    simd_float4 xi3 = simd_float4(
        float(X[(b + 12) * n_features + i]),
        float(X[(b + 13) * n_features + i]),
        float(X[(b + 14) * n_features + i]),
        float(X[(b + 15) * n_features + i])
    );
    simd_float4 xj3 = simd_float4(
        float(X[(b + 12) * n_features + j]),
        float(X[(b + 13) * n_features + j]),
        float(X[(b + 14) * n_features + j]),
        float(X[(b + 15) * n_features + j])
    );
    acc3 += xi3 * xj3;
}

// Reduce all accumulators manually
sum = acc0[0] + acc0[1] + acc0[2] + acc0[3] +
      acc1[0] + acc1[1] + acc1[2] + acc1[3] +
      acc2[0] + acc2[1] + acc2[2] + acc2[3] +
      acc3[0] + acc3[1] + acc3[2] + acc3[3];

// Process remaining 8 elements
if (b + 7 < batch_size) {
    simd_float4 xi0 = simd_float4(
        float(X[(b + 0) * n_features + i]),
        float(X[(b + 1) * n_features + i]),
        float(X[(b + 2) * n_features + i]),
        float(X[(b + 3) * n_features + i])
    );
    simd_float4 xj0 = simd_float4(
        float(X[(b + 0) * n_features + j]),
        float(X[(b + 1) * n_features + j]),
        float(X[(b + 2) * n_features + j]),
        float(X[(b + 3) * n_features + j])
    );
    simd_float4 prod0 = xi0 * xj0;
    simd_float4 xi1 = simd_float4(
        float(X[(b + 4) * n_features + i]),
        float(X[(b + 5) * n_features + i]),
        float(X[(b + 6) * n_features + i]),
        float(X[(b + 7) * n_features + i])
    );
    simd_float4 xj1 = simd_float4(
        float(X[(b + 4) * n_features + j]),
        float(X[(b + 5) * n_features + j]),
        float(X[(b + 6) * n_features + j]),
        float(X[(b + 7) * n_features + j])
    );
    simd_float4 prod1 = xi1 * xj1;
    sum += prod0[0] + prod0[1] + prod0[2] + prod0[3] + prod1[0] + prod1[1] + prod1[2] + prod1[3];
    b += 8;
}

// Process remaining 4 elements with 4-wide SIMD
if (b + 3 < batch_size) {
    simd_float4 xi = simd_float4(
        float(X[(b + 0) * n_features + i]),
        float(X[(b + 1) * n_features + i]),
        float(X[(b + 2) * n_features + i]),
        float(X[(b + 3) * n_features + i])
    );
    simd_float4 xj = simd_float4(
        float(X[(b + 0) * n_features + j]),
        float(X[(b + 1) * n_features + j]),
        float(X[(b + 2) * n_features + j]),
        float(X[(b + 3) * n_features + j])
    );
    simd_float4 prod = xi * xj;
    sum += prod[0] + prod[1] + prod[2] + prod[3];
    b += 4;
}

// Handle remaining elements
for (; b < batch_size; b++) {
    sum += float(X[b * n_features + i]) * float(X[b * n_features + j]);
}

out_data[tid] = bfloat16_t(sum / float(batch_size));
"""

masked_matmul_kernel = mx.fast.metal_kernel(
    name="masked_matmul",
    input_names=["a_rows", "a_cols", "a_data", "b_rows", "b_cols", "b_data",
                 "mask_rows", "mask_cols", "a_nnz", "b_nnz", "mask_nnz"],
    output_names=["out_data"],
    source=masked_matmul_source
)

sparse_vector_matmul_kernel = mx.fast.metal_kernel(
    name="sparse_vector_matmul",
    input_names=["mat_rows", "mat_cols", "mat_data", "vec", "nnz", "n_rows"],
    output_names=["out"],
    source=sparse_vector_matmul_source
)

masked_correlation_kernel = mx.fast.metal_kernel(
    name="masked_correlation",
    input_names=["X", "mask_rows", "mask_cols", "batch_size", "n_features", "mask_nnz"],
    output_names=["out_data"],
    source=masked_correlation_source
)

def _masked_matmul(
    a_rows: mx.array,
    a_cols: mx.array,
    a_data: mx.array,
    b_rows: mx.array,
    b_cols: mx.array,
    b_data: mx.array,
    mask_rows: mx.array,
    mask_cols: mx.array,
) -> mx.array:
    """
    Perform sparse matrix multiplication with mask: mask * (A @ B)

    Args:
        a_rows: Row indices of the first matrix
        a_cols: Column indices of the first matrix
        a_data: Non-zero values of the first matrix
        b_rows: Row indices of the second matrix
        b_cols: Column indices of the second matrix
        b_data: Non-zero values of the second matrix
        mask_rows: Row indices of the mask
        mask_cols: Column indices of the mask

    Returns:
        Non-zero values at mask positions
    """
    a_nnz = mx.array(a_data.shape[0], dtype=mx.int32)
    b_nnz = mx.array(b_data.shape[0], dtype=mx.int32)
    mask_nnz = mx.array(mask_rows.shape[0], dtype=mx.int32)
    
    # Use the dtype of the input data
    output_dtype = a_data.dtype

    outputs = masked_matmul_kernel( # pyright: ignore
            inputs=[a_rows, a_cols, a_data, b_rows, b_cols, b_data,
                   mask_rows, mask_cols, a_nnz, b_nnz, mask_nnz],
            grid=(mask_nnz.item(), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(mask_nnz.item(),)],
            output_dtypes=[output_dtype],
            stream=mx.gpu
    )
    return outputs[0]

def _sparse_vector_matmul(
    mat_rows: mx.array,
    mat_cols: mx.array,
    mat_data: mx.array,
    vec: mx.array,
    n_rows: int,
) -> mx.array:
    """
    Perform sparse matrix-vector multiplication: y = A @ x

    Args:
        mat_rows: Row indices of the sparse matrix
        mat_cols: Column indices of the sparse matrix
        mat_data: Non-zero values of the sparse matrix
        vec: Dense vector to multiply with
        n_rows: Number of rows in the output

    Returns:
        Dense result vector
    """
    nnz = mx.array(mat_data.shape[0], dtype=mx.int32)
    n_rows_array = mx.array(n_rows, dtype=mx.int32)
    
    # Use the dtype of the input data (prefer vec dtype if both are present)
    output_dtype = vec.dtype if vec.dtype == mx.bfloat16 else mat_data.dtype

    outputs = sparse_vector_matmul_kernel( # pyright: ignore
            inputs=[mat_rows, mat_cols, mat_data, vec, nnz, n_rows_array],
            grid=(n_rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n_rows,)],
            output_dtypes=[output_dtype],
            stream=mx.gpu
    )
    return outputs[0]

def _masked_correlation(
    X: mx.array,
    mask_rows: mx.array,
    mask_cols: mx.array,
) -> mx.array:
    """
    Compute masked correlation: mask * (X.T @ X) / batch_size

    Args:
        X: Dense matrix of shape (batch_size, n_features)
        mask_rows: Row indices where to compute correlation
        mask_cols: Column indices where to compute correlation

    Returns:
        Correlation values at mask positions
    """
    batch_size = mx.array(X.shape[0], dtype=mx.int32)
    n_features = mx.array(X.shape[1], dtype=mx.int32)
    mask_nnz = mx.array(mask_rows.shape[0], dtype=mx.int32)
    
    # Use the dtype of the input matrix X
    output_dtype = X.dtype

    outputs = masked_correlation_kernel( # pyright: ignore
            inputs=[X, mask_rows, mask_cols, batch_size, n_features, mask_nnz],
            grid=(mask_nnz.item(), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(mask_nnz.item(),)],
            output_dtypes=[output_dtype],
            stream=mx.gpu
    )
    return outputs[0]

class Matrix:
    """
    A sparse matrix representation using MLX arrays in COO (Coordinate) format.
    """

    def __init__(self, rows: mx.array, cols: mx.array, data: mx.array, shape: tuple, dtype: mx.Dtype = mx.bfloat16):
        """
        Initialize a sparse matrix in COO format.

        Args:
            rows: Row indices of non-zero values
            cols: Column indices of non-zero values
            data: The non-zero values of the matrix
            shape: The shape of the matrix (n_rows, n_cols)
            dtype: Data type for the values
        """
        self.shape = shape
        self.rows = rows.astype(mx.int32)
        self.cols = cols.astype(mx.int32)
        self.data = data.astype(dtype)
        self.dtype = dtype
        self.nnz = data.shape[0]

    def to_dense(self) -> mx.array:
        """Convert sparse matrix to dense format (compile-safe version).

        This version works in both regular and compiled contexts by avoiding .item() calls.
        Instead, it uses MLX's built-in indexing with integer arrays.
        """
        dense = mx.zeros(self.shape, dtype=self.dtype)

        # MLX supports indexing with integer arrays directly
        # Create linear indices: idx = row * ncols + col
        linear_indices = self.rows * self.shape[1] + self.cols

        # Flatten, scatter values, then reshape
        flat = dense.reshape(-1)
        # Use MLX's array indexing (no .item() needed!)
        flat = flat.at[linear_indices].add(self.data)
        dense = flat.reshape(self.shape)

        return dense

    def masked_matmul(self, other: 'Matrix', mask: 'Matrix') -> 'Matrix':
        """
        Matrix multiplication with another matrix, masked by a third matrix.
        Computes mask * (self @ other) efficiently by only computing at mask positions.

        Args:
            other: Another sparse matrix
            mask: Mask matrix (only positions with non-zeros will be computed)

        Returns:
            Sparse matrix with values at mask positions
        """
        out_data = _masked_matmul(
            self.rows, self.cols, self.data,
            other.rows, other.cols, other.data,
            mask.rows, mask.cols
        )
        return Matrix(mask.rows, mask.cols, out_data, mask.shape, self.dtype)

    def __matmul__(self, other):
        """Matrix multiplication operator (@)."""
        if isinstance(other, Matrix):
            # For sparse @ sparse without mask, convert to dense
            # (masked version is more efficient when you have a mask)
            return self.to_dense() @ other.to_dense()
        elif isinstance(other, mx.array):
            if len(other.shape) == 1:
                # Sparse matrix @ dense vector
                return _sparse_vector_matmul(
                    self.rows, self.cols, self.data, other, self.shape[0]
                )
            else:
                # Sparse matrix @ dense matrix
                return self.to_dense() @ other
        else:
            return NotImplemented

    def __mul__(self, other):
        """Element-wise multiplication (masking)."""
        if isinstance(other, Matrix):
            # Element-wise multiply (masking): keep only values where both are non-zero
            mask_set = set(zip(other.rows.tolist(), other.cols.tolist()))
            new_rows, new_cols, new_data = [], [], []

            for i in range(self.nnz):
                r, c = self.rows[i].item(), self.cols[i].item()
                if (r, c) in mask_set:
                    # Find the corresponding value in other
                    idx = next((j for j in range(other.nnz)
                              if other.rows[j].item() == r and other.cols[j].item() == c), None)
                    if idx is not None:
                        new_rows.append(r)
                        new_cols.append(c)
                        new_data.append(self.data[i].item() * other.data[idx].item())

            if len(new_data) == 0:
                return Matrix(mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32),
                            mx.array([], dtype=self.dtype), self.shape, self.dtype)

            return Matrix(mx.array(new_rows, dtype=mx.int32), mx.array(new_cols, dtype=mx.int32),
                        mx.array(new_data, dtype=self.dtype), self.shape, self.dtype)
        elif isinstance(other, (int, float, mx.array)):
            # Scalar multiplication
            return Matrix(self.rows, self.cols, self.data * other, self.shape, self.dtype)
        else:
            return NotImplemented


def from_graph(graph, dtype=mx.bfloat16) -> Matrix:
    """
    Create a sparse matrix from a NetworkX graph.

    Args:
        graph: A NetworkX graph object
        dtype: Data type for the matrix values

    Returns:
        A sparse Matrix object
    """
    np_graph = nx.to_numpy_array(graph)
    shape = np_graph.shape

    rows, cols = np.where(np_graph != 0)
    data = np_graph[rows, cols]

    rows = mx.array(rows, dtype=mx.int32)
    cols = mx.array(cols, dtype=mx.int32)
    data = mx.array(data, dtype=dtype)

    return Matrix(rows, cols, data, shape, dtype)

def from_dense(dense: mx.array, dtype=None) -> Matrix:
    """
    Create a sparse matrix from a dense MLX array.

    Args:
        dense: Dense MLX array
        dtype: Data type (if None, uses dense array's dtype)

    Returns:
        A sparse Matrix object
    """
    if dtype is None:
        dtype = dense.dtype

    # Convert to numpy for easier indexing
    # bfloat16 is not directly supported by numpy, so convert to float32 first
    if dense.dtype == mx.bfloat16:
        dense_for_numpy = dense.astype(mx.float32)
    else:
        dense_for_numpy = dense
    np_dense = np.array(dense_for_numpy)
    rows, cols = np.where(np_dense != 0)
    data = np_dense[rows, cols]

    rows = mx.array(rows, dtype=mx.int32)
    cols = mx.array(cols, dtype=mx.int32)
    data = mx.array(data, dtype=dtype)

    return Matrix(rows, cols, data, dense.shape, dtype)

def masked_correlation(X: mx.array, mask: Matrix) -> Matrix:
    """
    Compute masked correlation: mask * (X.T @ X) / batch_size

    This efficiently computes the correlation matrix only at positions where mask is non-zero.

    Args:
        X: Dense matrix of shape (batch_size, n_features)
        mask: Sparse mask matrix indicating which correlations to compute

    Returns:
        Sparse matrix containing correlation values at mask positions
    """
    out_data = _masked_correlation(X, mask.rows, mask.cols)
    return Matrix(mask.rows, mask.cols, out_data, mask.shape, X.dtype)
