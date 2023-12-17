#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace cl::sycl;

void convolve(queue& q, buffer<float, 1>& buffer_input, buffer<float, 1>& buffer_kernel,
              buffer<float, 1>& buffer_output, int input_rows, int input_cols,
              int kernel_rows, int kernel_cols, int output_rows, int output_cols) {
    // Submit command group to queue
    q.submit([&](handler& h) {
        // Get accessors for input, kernel, and output
        auto acc_input = buffer_input.get_access<access::mode::read>(h);
        auto acc_kernel = buffer_kernel.get_access<access::mode::read>(h);
        auto acc_output = buffer_output.get_access<access::mode::write>(h);

        // Define the kernel
        h.parallel_for(range<1>(output_rows * output_cols), [=](id<1> idx) {
            int output_index = idx[0];
            int row = output_index / output_cols;
            int col = output_index % output_cols;
            float sum = 0.0f;

            for (int i = 0; i < kernel_rows; ++i) {
                for (int j = 0; j < kernel_cols; ++j) {
                    sum += acc_input[(row + i) * input_cols + (col + j)] * acc_kernel[i * kernel_cols + j];
                }
            }

            acc_output[output_index] = sum;
        });
    });
}
// Interface function for Python to call
extern "C" void convolve_interface(const float* input_matrix_data, const float* kernel_data, float* output_matrix_data,
                                   int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // Define the dimensions for the output matrix
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Create SYCL queue
    queue q;

    // Create SYCL buffers from the raw pointers provided by Python
    buffer<float, 1> buffer_input(input_matrix_data, range<1>(input_rows * input_cols));
    buffer<float, 1> buffer_kernel(kernel_data, range<1>(kernel_rows * kernel_cols));
    buffer<float, 1> buffer_output(output_matrix_data, range<1>(output_rows * output_cols));

    // Perform the convolution
    convolve(q, buffer_input, buffer_kernel, buffer_output, input_rows, input_cols, kernel_rows, kernel_cols, output_rows, output_cols);

    // Wait for the queue to finish
    q.wait();
}
