import mlx.core as mx
import numpy as np

sparse_vector_matmul_source = """
// Sparse matrix-vector multiplication: y = A @ x
// A is in CSR format (row_ptr, cols, data)
// x is a dense vector
uint tid = thread_position_in_grid.x;
if (tid >= n_rows) return;

bfloat16_t sum = bfloat16_t(0);

// CSR: direct lookup of row start and end
int row_start = row_ptr[tid];
int row_end = row_ptr[tid + 1];

// Iterate through all entries in this row
for (int idx = row_start; idx < row_end; idx++) {
    sum += mat_data[idx] * vec[mat_cols[idx]];
}

out[tid] = sum;
"""

sparse_matrix_matmul_source = """
// Sparse matrix-dense matrix multiplication: Y = A @ X
// A is in CSR format (row_ptr, cols, data) - shape (n_rows, n_cols_A)
// X is a dense matrix - shape (n_cols_A, n_cols_X)
// Y is output dense matrix - shape (n_rows, n_cols_X)
uint row_tid = thread_position_in_grid.x;
uint col_tid = thread_position_in_grid.y;

if (row_tid >= n_rows || col_tid >= n_cols_out) return;

bfloat16_t sum = bfloat16_t(0);

// CSR: direct lookup of row start and end
int row_start = row_ptr[row_tid];
int row_end = row_ptr[row_tid + 1];

// Iterate through all entries in this row
for (int idx = row_start; idx < row_end; idx++) {
    int k = mat_cols[idx];
    // X is stored in row-major: X[k, col_tid] = X[k * n_cols_out + col_tid]
    sum += mat_data[idx] * dense_mat[k * n_cols_out + col_tid];
}

out[row_tid * n_cols_out + col_tid] = sum;
"""

masked_correlation_source = """
// Compute masked correlation: mask * (X.T @ X) / batch_size
// X is (batch_size, n_features), result is sparse where mask is non-zero
// mask is in COO format (mask_rows, mask_cols) for efficient element-wise access
uint tid = thread_position_in_grid.x;
if (tid >= mask_nnz) return;

int row_i = mask_rows[tid];
int col_j = mask_cols[tid];

bfloat16_t sum = bfloat16_t(0);

// Compute dot product of column row_i and column col_j
for (int b = 0; b < batch_size; b++) {
    sum += X[b * n_features + row_i] * X[b * n_features + col_j];
}

out_data[tid] = sum / bfloat16_t(batch_size);
"""

sparse_vector_matmul_kernel = mx.fast.metal_kernel(
    name="sparse_vector_matmul",
    input_names=["row_ptr", "mat_cols", "mat_data", "vec", "n_rows"],
    output_names=["out"],
    source=sparse_vector_matmul_source,
)

sparse_matrix_matmul_kernel = mx.fast.metal_kernel(
    name="sparse_matrix_matmul",
    input_names=[
        "row_ptr",
        "mat_cols",
        "mat_data",
        "dense_mat",
        "n_rows",
        "n_cols_out",
    ],
    output_names=["out"],
    source=sparse_matrix_matmul_source,
)

masked_correlation_kernel = mx.fast.metal_kernel(
    name="masked_correlation",
    input_names=[
        "X",
        "mask_rows",
        "mask_cols",
        "batch_size",
        "n_features",
        "mask_nnz",
    ],
    output_names=["out_data"],
    source=masked_correlation_source,
)


def _sparse_vector_matmul(
    row_ptr: mx.array,
    mat_cols: mx.array,
    mat_data: mx.array,
    vec: mx.array,
    n_rows: int,
) -> mx.array:
    """
    Perform sparse matrix-vector multiplication: y = A @ x

    Args:
        row_ptr: CSR row pointer array (size n_rows + 1)
        mat_cols: Column indices of the sparse matrix
        mat_data: Non-zero values of the sparse matrix
        vec: Dense vector to multiply with
        n_rows: Number of rows in the output

    Returns:
        Dense result vector
    """
    n_rows_array = mx.array(n_rows, dtype=mx.int32)

    # Use the dtype of the input data (prefer vec dtype if both are present)
    output_dtype = vec.dtype if vec.dtype == mx.bfloat16 else mat_data.dtype

    outputs = sparse_vector_matmul_kernel(  # pyright: ignore
        inputs=[row_ptr, mat_cols, mat_data, vec, n_rows_array],
        grid=(n_rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_rows,)],
        output_dtypes=[output_dtype],
        stream=mx.gpu,
    )
    return outputs[0]


def _sparse_matrix_matmul(
    row_ptr: mx.array,
    mat_cols: mx.array,
    mat_data: mx.array,
    dense_mat: mx.array,
    n_rows: int,
) -> mx.array:
    """
    Perform sparse matrix-dense matrix multiplication: Y = A @ X

    Args:
        row_ptr: CSR row pointer array (size n_rows + 1)
        mat_cols: Column indices of the sparse matrix
        mat_data: Non-zero values of the sparse matrix
        dense_mat: Dense matrix to multiply with (n_cols_A, n_cols_X)
        n_rows: Number of rows in the sparse matrix

    Returns:
        Dense result matrix (n_rows, n_cols_X)
    """
    n_rows_array = mx.array(n_rows, dtype=mx.int32)
    n_cols_out = dense_mat.shape[1]
    n_cols_out_array = mx.array(n_cols_out, dtype=mx.int32)

    # Use the dtype of the input data (prefer dense_mat dtype if both are present)
    output_dtype = dense_mat.dtype if dense_mat.dtype == mx.bfloat16 else mat_data.dtype

    outputs = sparse_matrix_matmul_kernel(  # pyright: ignore
        inputs=[row_ptr, mat_cols, mat_data, dense_mat, n_rows_array, n_cols_out_array],
        grid=(n_rows, n_cols_out, 1),
        threadgroup=(16, 16, 1),
        output_shapes=[(n_rows * n_cols_out,)],
        output_dtypes=[output_dtype],
        stream=mx.gpu,
    )
    return outputs[0].reshape(n_rows, n_cols_out)


def _masked_correlation(
    X: mx.array,
    mask_rows: mx.array,
    mask_cols: mx.array,
) -> mx.array:
    """
    Compute masked correlation: mask * (X.T @ X) / batch_size

    Args:
        X: Dense matrix of shape (batch_size, n_features)
        mask_rows: COO row indices for the mask
        mask_cols: Column indices where to compute correlation

    Returns:
        Correlation values at mask positions
    """
    batch_size = mx.array(X.shape[0], dtype=mx.int32)
    n_features = mx.array(X.shape[1], dtype=mx.int32)
    mask_nnz = mx.array(mask_cols.shape[0], dtype=mx.int32)

    # Use the dtype of the input matrix X
    output_dtype = X.dtype

    outputs = masked_correlation_kernel(  # pyright: ignore
        inputs=[X, mask_rows, mask_cols, batch_size, n_features, mask_nnz],
        grid=(mask_nnz.item(), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(mask_nnz.item(),)],
        output_dtypes=[output_dtype],
        stream=mx.gpu,
    )
    return outputs[0]


class Matrix:
    """
    A sparse matrix representation using MLX arrays in hybrid COO/CSR format.
    Stores both row indices (COO) and row pointers (CSR) for optimal kernel performance.
    """

    def __init__(
        self,
        row_ptr: mx.array,
        rows: mx.array,
        cols: mx.array,
        data: mx.array,
        shape: tuple,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        """
        Initialize a sparse matrix in hybrid COO/CSR format.

        Args:
            row_ptr: CSR row pointer array (size n_rows + 1)
            rows: COO row indices of non-zero values
            cols: Column indices of non-zero values
            data: The non-zero values of the matrix
            shape: The shape of the matrix (n_rows, n_cols)
            dtype: Data type for the values
        """
        self.shape = shape
        self.dtype = dtype
        self.row_ptr = row_ptr.astype(mx.int32)
        self.rows = rows.astype(mx.int32)
        self.cols = cols.astype(mx.int32)
        self.data = data.astype(dtype)
        self.nnz = data.shape[0]

    def __matmul__(self, other):
        """Matrix multiplication operator (@)."""
        if isinstance(other, mx.array):
            if len(other.shape) == 1:
                # Sparse matrix @ dense vector
                return _sparse_vector_matmul(
                    self.row_ptr, self.cols, self.data, other, self.shape[0]
                )
            else:
                # Sparse matrix @ dense matrix - use custom kernel
                return _sparse_matrix_matmul(
                    self.row_ptr, self.cols, self.data, other, self.shape[0]
                )
        else:
            return NotImplemented


def from_dense(dense: mx.array, dtype=None) -> Matrix:
    """
    Create a sparse matrix from a dense MLX array.

    Args:
        dense: Dense MLX array
        dtype: Data type (if None, uses dense array's dtype)

    Returns:
        A sparse Matrix object in hybrid COO/CSR format
    """
    if dtype is None:
        dtype = dense.dtype

    # Convert to numpy for easier indexing
    # bfloat16 is not directly supported by numpy, so convert to float32 first
    dense_for_numpy = dense.astype(mx.float32)
    np_dense = np.array(dense_for_numpy)
    rows, cols = np.where(np_dense != 0)
    data = np_dense[rows, cols]

    # Sort by row, then by column for CSR format
    sort_indices = np.lexsort((cols, rows))
    rows = rows[sort_indices]
    cols = cols[sort_indices]
    data = data[sort_indices]

    # Build CSR row_ptr array
    row_ptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    for i in range(len(rows)):
        row_ptr[rows[i] + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    # Convert to MLX arrays
    row_ptr = mx.array(row_ptr, dtype=mx.int32)
    rows = mx.array(rows, dtype=mx.int32)  # Keep COO row indices
    cols = mx.array(cols, dtype=mx.int32)
    data = mx.array(data, dtype=dtype)

    return Matrix(row_ptr, rows, cols, data, dense.shape, dtype)


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
    return Matrix(mask.row_ptr, mask.rows, mask.cols, out_data, mask.shape, X.dtype)
