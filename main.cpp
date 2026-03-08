#include <iostream>
#include <chrono>
#include "matrix.h"

int main()
{
    // Create 2x3 matrix
    matrix<double> A(1000, 1000, true);

    // std::cout << "Matrix A:\n";
    //  A.print();

    // Create 3x2 matrix
    matrix<double> B(1000, 1000, true);

    // std::cout << "Matrix B:\n";
    // B.print();

    // Multiply B * A? Make sure dimensions align!
    // B is 3x2, A is 2x3 → B*A = 3x3
    auto start = std::chrono::high_resolution_clock::now();
    matrix<double> C = B.matmul(A);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    // Output the latency
    std::cout << "Latency: " << duration.count() << " ns" << std::endl;

    // std::cout << "Matrix C = B * A:\n";
    // C.print();

    start = std::chrono::high_resolution_clock::now();
    matrix<double> multi_thread_C = B.matmul_multithreaded(A);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    // Output the latency
    std::cout << "Latency: " << duration.count() << " ns" << std::endl;

    // std::cout << "Matrix C = B * A:\n";
    // C.print();

    return 0;
}
