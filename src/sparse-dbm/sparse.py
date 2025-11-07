import mlx.core as mx
import numpy as np
import networkx as nx

masked_matmul_source = """
// Sparse matrix-matrix multiplication with mask
// This kernel computes C = mask * (A @ B) where A and B are in COO format
uint tid = thread_position_in_grid.x;
if (tid >= mask_nnz) return;

int i = mask_rows[tid];
int j = mask_cols[tid];

bfloat16_t sum = bfloat16_t(0);

// For each non-zero in A's row i
for (int a_idx = 0; a_idx < a_nnz; a_idx++) {
    if (a_rows[a_idx] == i) {
        int k = a_cols[a_idx];
        bfloat16_t a_val = a_data[a_idx];

        // Find matching column in B (k-th row of B)
        for (int b_idx = 0; b_idx < b_nnz; b_idx++) {
            if (b_rows[b_idx] == k && b_cols[b_idx] == j) {
                sum += a_val * b_data[b_idx];
            }
        }
    }
}

out_data[tid] = sum;
"""

sparse_vector_matmul_source = """
// Sparse matrix-vector multiplication: y = A @ x
// A is in COO format (rows, cols, data)
// x is a dense vector
uint tid = thread_position_in_grid.x;
if (tid >= n_rows) return;

bfloat16_t sum = bfloat16_t(0);

// Sum all non-zeros in this row
for (int i = 0; i < nnz; i++) {
    if (mat_rows[i] == int(tid)) {
        sum += mat_data[i] * vec[mat_cols[i]];
    }
}

out[tid] = sum;
"""

masked_correlation_source = """
// Compute masked correlation: mask * (X.T @ X) / batch_size
// X is (batch_size, n_features), result is sparse where mask is non-zero
uint tid = thread_position_in_grid.x;
if (tid >= mask_nnz) return;

int i = mask_rows[tid];
int j = mask_cols[tid];

bfloat16_t sum = bfloat16_t(0);

// Compute dot product of column i and column j
for (int b = 0; b < batch_size; b++) {
    sum += X[b * n_features + i] * X[b * n_features + j];
}

out_data[tid] = sum / bfloat16_t(batch_size);
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
        """Convert sparse matrix to dense format."""
        dense = mx.zeros(self.shape, dtype=self.dtype)
        # Use scatter to efficiently set values
        for i in range(self.nnz):
            dense[self.rows[i].item(), self.cols[i].item()] = self.data[i]
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
