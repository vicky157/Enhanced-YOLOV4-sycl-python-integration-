import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libsmm.so')

# Defined the types for the C++ function arguments and return type
lib.convolve_interface.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                   ctypes.POINTER(ctypes.c_float),
                                   ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int, 
                                   ctypes.c_int, 
                                   ctypes.c_int, 
                                   ctypes.c_int]
lib.convolve_interface.restype = None

def convolve(input_matrix, kernel):
    # Ensure the input is a flat, contiguous array of floats
    input_matrix = np.ascontiguousarray(input_matrix, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    output_rows = input_matrix.shape[0] - kernel.shape[0] + 1
    output_cols = input_matrix.shape[1] - kernel.shape[1] + 1
    output_matrix = np.empty((output_rows, output_cols), dtype=np.float32)

    # Call the C++ convolve_interface function
    lib.convolve_interface(input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           output_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           input_matrix.shape[0], input_matrix.shape[1],
                           kernel.shape[0], kernel.shape[1])

    return output_matrix

# Example usage
if __name__ == "__main__":
    input_matrix = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]], dtype=np.float32)

    kernel = np.array([[1, 2],
                       [3, 4]], dtype=np.float32)

    result = convolve(input_matrix, kernel)
    print("Convolution Result:")
    print(result)
