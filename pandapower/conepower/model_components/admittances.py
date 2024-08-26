import numpy as np
from scipy import sparse


class Admittances:
    y_bus: sparse.csr_matrix
    y_ff: np.ndarray
    y_ft: np.ndarray
    y_tf: np.ndarray
    y_tt: np.ndarray

    def __init__(self,
                 y_bus: sparse.csc_matrix,
                 y_ff: np.ndarray,
                 y_ft: np.ndarray,
                 y_tf: np.ndarray,
                 y_tt: np.ndarray):
        self.y_bus = y_bus.tocsr()
        self.y_ff = y_ff
        self.y_ft = y_ft
        self.y_tf = y_tf
        self.y_tt = y_tt
