#ifndef NEW_H
#define NEW_H

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

// Convolution function
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

// int main() {
//     // Define input matrix and kernel
//     std::vector<float> input_matrix = {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//     };

//     std::vector<float> kernel = {
//         1, 2,
//         3, 4
//     };

//     const int input_rows = 4;
//     const int input_cols = 4;
//     const int kernel_rows = 2;
//     const int kernel_cols = 2;
//     const int output_rows = input_rows - kernel_rows + 1;
//     const int output_cols = input_cols - kernel_cols + 1;

//     std::vector<float> output_matrix(output_rows * output_cols, 0);

//     // Create buffers
//     buffer<float, 1> buffer_input(input_matrix.data(), range<1>(input_matrix.size()));
//     buffer<float, 1> buffer_kernel(kernel.data(), range<1>(kernel.size()));
//     buffer<float, 1> buffer_output(output_matrix.data(), range<1>(output_matrix.size()));

//     // Initialize SYCL queue
//     queue q;
//     auto start = std::chrono::steady_clock::now();
//     // Perform Convolution
//     convolve(q, buffer_input, buffer_kernel, buffer_output, input_rows, input_cols, kernel_rows, kernel_cols, output_rows, output_cols);

//     // Wait for the queue to finish
//     q.wait();
//     auto end = std::chrono::steady_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Time taken for convolution: " << duration.count() << " seconds." << std::endl;

//     // Print the output matrix
//     auto host_output = buffer_output.get_access<access::mode::read>();
//     for (int i = 0; i < output_rows; ++i) {
//         for (int j = 0; j < output_cols; ++j) {
//             std::cout << host_output[i * output_cols + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }


#endif // NEW_H