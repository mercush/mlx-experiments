import mlx.core as mx
import numpy as np
import networkx as nx

masked_matmul_source = """
"""

dense_vector_matmul_source = """
"""

masked_matmul_kernel = mx.fast.metal_kernel(
    name="masked_matmul",
    input_names=["inp"],
    output_names=["out"],
    source=masked_matmul_source
)

dense_vector_matmul_kernel = mx.fast.metal_kernel(
    name="vector_matmul",
    input_names=["inp"],
    output_names=["out"],
    source=masked_matmul_source
)

def _masked_matmul(
    a_rows: mx.array,
    a_cols: mx.array,
    a_data: mx.array,
    b_rows: mx.array,
    b_cols: mx.array,
    b_data: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Perform sparse matrix multiplication between two matrices in CSR format.
    
    Args:
        a_rows: Row indices of the first matrix
        a_cols: Column indices of the first matrix
        a_data: Non-zero values of the first matrix
        b_rows: Row indices of the second matrix
        b_cols: Column indices of the second matrix
        b_data: Non-zero values of the second matrix
        
    Returns:
        A tuple of (rows, cols, data) arrays representing the result matrix
    """
    
    outputs = masked_matmul_kernel( # pyright: ignore
            inputs=[a_rows, a_cols, a_data, b_rows, b_cols, b_data],
            grid=mx.array([1]),
            threadgroup=mx.array([1]),
            output_shapes=[(1,), (1,), (1,)],
            output_dtypes=[mx.float32, mx.int32, mx.int32]
    )
    return outputs

def _vector_matmul(
    vec: mx.array,
    mat_rows: mx.array,
    mat_cols: mx.array,
    mat_data: mx.array,
) -> mx.array:
    """
    Perform sparse matrix-vector multiplication.
    
    Args:
        vec: Dense vector to multiply with
        mat_rows: Row indices of the sparse matrix
        mat_cols: Column indices of the sparse matrix
        mat_data: Non-zero values of the sparse matrix
        
    Returns:
        Dense result vector
    """
    
    outputs = dense_vector_matmul_kernel( # pyright: ignore
            inputs=[vec, mat_rows, mat_cols, mat_data],
            grid=mx.array([1]),
            threadgroup=mx.array([1]),
            output_shapes=[(vec.shape[0],)],
            output_dtypes=[mx.float32]
    )
    return outputs[0]

class Matrix:
    """
    A sparse matrix representation using MLX arrays.
    """

    def __init__(self, rows: mx.array, cols: mx.array, data: mx.array, shape: tuple, dtype: mx.Dtype = mx.float32):
        """
        Initialize a sparse matrix.

        Args:
            data: The non-zero values of the matrix
            indices: The indices of the non-zero values
            shape: The shape of the matrix
        """
        self.shape = shape
        self.rows = rows
        self.cols = cols
        self.data = data
        self.dtype = dtype

    def to_dense(self) -> mx.array:
        """Convert sparse matrix to dense format."""
        dense = mx.zeros(self.shape, dtype=self.dtype)
        dense[self.rows, self.cols] = self.data
        return dense

    def masked_matmul(self, other, mask):
        """Matrix multiplication with another matrix or vector."""
        res = _masked_matmul(
            self.rows, self.cols, self.data, other.rows, other.cols, other.data
        )
        return Matrix(res[0], res[1], res[2], mask.shape, self.dtype)

    def vector_matmul(self, vector):
        """Vector multiplication with a dense vector."""
        res = _vector_matmul(vector, self.rows, self.cols, self.data)
        return res


def from_graph(graph, dtype=mx.float32) -> Matrix:
    """
    Create a sparse matrix from an edge list.

    Args:
        edgelist: A list of tuples (row, col, value) representing non-zero entries

    Returns:
        A sparse Matrix object
    """
    np_graph = nx.to_numpy_array(graph)
    shape = np_graph.shape
    
    rows, cols = np.where(np_graph != 0)
    data = np_graph[rows, cols]
    
    rows = mx.array(rows)
    cols = mx.array(cols)
    data = mx.array(data, dtype=dtype)
    
    return Matrix(rows, cols, data, shape, dtype)
