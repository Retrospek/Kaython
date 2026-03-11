#include "matrix.h"
#include <algorithm>
#include <cmath>

template <typename TD>
class Activations
{
private:
    template <typename Func>
    void _parallel_operator_overload(Func func, size_t row_num) const
    {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        num_threads = std::min(num_threads, row_num);

        size_t rows_per_thread = row_num / num_threads;
        size_t r_start = 0;

        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (size_t i = 0; i < num_threads; ++i)
        {
            size_t r_end = (i == num_threads - 1) ? row_num : r_start + rows_per_thread;
            threads.emplace_back(func, r_start, r_end);
            r_start = r_end;
        }
    }

public:
    static TD relu(matrix<TD> &data)
    {
        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    { 
            for (size_t r = row_start; r < row_end; ++r) {
                for (size_t c{0}; c < data.get_col_num(); ++c) {
                    data.at(r, c) = std::max(0, data.at(r, c));
                }
        } }, data.get_row_num());
    }
    static TD relu_grad(matrix<TD> &data)
    {
        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    { 
            for (size_t r = row_start; r < row_end; ++r) {
                for (size_t c{0}; c < data.get_col_num(); ++c) {
                    data.at(r, c) = std::max(0, data.at(r, c));
                }
        } }, data.get_row_num());
    }
};