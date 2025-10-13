import mlx.core as mx

class Matrix:
    """
    A sparse matrix representation using MLX arrays.
    """
    
    def __init__(self, rows: mx.array, cols: mx.array, data: mx.array, shape: tuple):
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
        
    def to_dense(self) -> mx.array:
        """Convert sparse matrix to dense format."""
        dense = mx.zeros(self.shape)
        dense[self.rows, self.cols] = self.data
        return dense
        
    @mx.compile
    def __matmul__(self, other):
        """Matrix multiplication with another matrix or vector."""

        result_rows = []
        result_cols = []
        result_data = []
        
        for i in range(len(self.rows)):
            row_idx = self.rows[i]
            col_idx = self.cols[i]
            val = self.data[i]
            
            for j in range(len(other.rows)):
                if other.rows[j] == col_idx:
                    result_row = row_idx
                    result_col = other.cols[j]
                    result_val = val * other.data[j]
                    
                    found = False
                    for k in range(len(result_rows)):
                        if result_rows[k] == result_row and result_cols[k] == result_col:
                            result_data[k] += result_val
                            found = True
                            break
                    
                    if not found:
                        result_rows.append(result_row)
                        result_cols.append(result_col)
                        result_data.append(result_val)
        
        return Matrix(
            mx.array(result_rows),
            mx.array(result_cols),
            mx.array(result_data),
            (self.shape[0], other.shape[1])
        )

def from_edgelist(edgelist) -> Matrix:
    """
    Create a sparse matrix from an edge list.
    
    Args:
        edgelist: A list of tuples (row, col, value) representing non-zero entries
        
    Returns:
        A sparse Matrix object
    """
    if not edgelist:
        return Matrix(mx.array([]), mx.array([]), mx.array([]), (0, 0))
    
    rows, cols, data = zip(*edgelist)
    
    # Determine matrix shape from max indices
    max_row = max(rows) if rows else 0
    max_col = max(cols) if cols else 0
    shape = (max_row + 1, max_col + 1)
    
    return Matrix(mx.array(rows), mx.array(cols), mx.array(data), shape)

