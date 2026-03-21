#include <vector>
#include <random>
#include <thread>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <functional>

template <typename DT>
class matrix
{
private:
    std::vector<DT> _data;
    size_t _row_num;
    size_t _col_num;

    inline static std::mt19937 rng{42}; // standard seed=42, we use {} instead of () because it lets the parser know we are doing initiailization and not a function call
    inline static std::uniform_real_distribution<DT> dist{0.0, 10.0};
    void random_vec(std::vector<DT> &curr, size_t row_num, size_t col_num)
    {
        curr.resize(row_num * col_num); // We know the exact amount of elements, so let's premt resize
        for (auto &val : curr)          // we can access the aliases instead of
        {
            val = dist(rng);
        }

        // By the end the vector should have ranodm values in the range 0-10 inclusive
    }

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
    /*
    6 things:
    - Destructor
    - Default Constructor
    - Copy Constructor
    - Copy Assignement
    - Move Constructor
    - Move Assignment
    */
    ~matrix() = default;
    matrix()
        : _data(), _row_num(0), _col_num(0)
    {
        /*
        By default a 0x0 matrix is created
        */
    }

    matrix(size_t row_num, size_t col_num, bool random)
        : _row_num(row_num), _col_num(col_num)
    {
        if (row_num > 0 && col_num > 0)
        {
            _data.resize(row_num * col_num);
            if (random)
                random_vec(_data, row_num, col_num);
        }
    }

    matrix(const matrix &other)
        : _data(other._data), _row_num(other._row_num), _col_num(other._col_num)
    {
    }

    matrix &operator=(const matrix &other)
    {
        if (this == &other) // if we're referencing the SAME object why the heck would we make a new object
        {
            return *this;
        }

        _data = other._data;
        _row_num = other._row_num;
        _col_num = other._col_num;

        return *this;
    }

    matrix(matrix &&other) noexcept // this will never through an exception because the rvalue is valid ALWAYS
        : _data(std::move(other._data)), _row_num(other._row_num), _col_num(other._col_num)
    {
        other._row_num = 0;
        other._col_num = 0;
    }

    matrix &operator=(matrix &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        _data = std::move(other._data);
        _row_num = other._row_num;
        _col_num = other._col_num;

        other._row_num = 0;
        other._col_num = 0;

        return *this;
    }

    std::vector<DT> &get_data()
    {
        return _data;
    }

    const std::vector<DT> &get_data() const // lowkey only useful when you are extracting a matrix's data to a third part thing
    {
        return _data;
    }

    size_t get_row_num() const
    {
        return _row_num;
    }

    size_t get_col_num() const
    {
        return _col_num;
    }

    void push_row(const std::vector<DT> &insert_row)
    {
        if (_col_num == 0) // So if we're dealing with an empty matrix just alter the col amount since rows change as you keep pushing but col -< constant
        {
            _col_num = insert_row.size();
        }
        else if (insert_row.size() != _col_num)
        {
            throw std::invalid_argument("Row size does not match matrix columns");
        }
        _data.insert(_data.end(), insert_row.begin(), insert_row.end()); // let's use the iterator that std::vector has implemented via begin and end
        ++_row_num;                                                      // Increase row count
    }

    void clear_data()
    {
        _data.clear();
    }

    DT &at(size_t row, size_t col) { return _data[row * _col_num + col]; }
    const DT &at(size_t row, size_t col) const { return _data[row * _col_num + col]; }

    matrix T() const // transpose copy
    {

        matrix<DT> new_mat(this->_col_num, this->_row_num, false);
        _parallel_operator_overload([&](size_t r_start, size_t r_end)
                                    {
        for (size_t r = r_start; r < r_end; ++r)
            for (size_t c = 0; c < this->_col_num; ++c)
                new_mat.at(c, r) = this->at(r, c); }, this->_row_num);

        return new_mat;
    }

    void T() // transpose reference
    {
        matrix<DT> new_mat(this->_col_num, this->_row_num, false);
        _parallel_operator_overload([&](size_t r_start, size_t r_end)
                                    {
        for (size_t r = r_start; r < r_end; ++r)
            for (size_t c = 0; c < this->_col_num; ++c)
                new_mat.at(c, r) = this->at(r, c); }, this->_row_num);

        *this = std::move(new_mat);
    }

    matrix inverse() const // Returns new matrix, leaves original unchanged
    {
        // Create augmented matrix [A | I]
        // Do Gaussian elimination
        // Extract and return the inverse

        if (_row_num != _col_num)
        {
            throw std::invalid_argument("yo send in a square matrix for an inversion operation R nxn not R nxm, where n != m");
        }

        DT determinant;
        matrix<DT> new_mat(_row_num, _col_num, false);
        if (_row_num == 1)
        {
            new_mat.at(0, 0) = 1.0 / this->at(0, 0);
        }
        else if (_row_num == 2)
        {
            // handle 2x2
            determinant = this->at(0, 0) * this->at(1, 1) - this->at(0, 1) * this->at(1, 0);
            new_mat.at(0, 0) = this->at(1, 1);
            new_mat.at(0, 1) = -this->at(0, 1);
            new_mat.at(1, 0) = -this->at(1, 0);
            new_mat.at(1, 1) = this->at(0, 0);
            new_mat *= (1.0 / determinant);
            return new_mat;
        }
        else if (_row_num == 3)
        {
            // handle 3x3
        }
        else
        {
            // general Gaussian elimination
        }
    }

    void inverse() // Modifies this matrix in-place
    {
        // Create augmented matrix [A | I]
        // Do Gaussian elimination
        // Extract the inverse
        // *this = std::move(result);
    }

    matrix matmul(const matrix &other) // left is = OTHER -> l_c = r_r
    {
        if (other._col_num != this->_row_num)
        {
            throw std::invalid_argument("Calling Other x Curr => Other Column Num != This Row Num");
        }

        matrix<DT> new_mat(other._row_num, _col_num, false);
        new_mat.clear_data();

        for (size_t r = 0; r < other._row_num; ++r) // LEFT MATRIX IS = OTHER
        {
            for (size_t c = 0; c < _col_num; ++c) // RIGHT MATRIX IS = THIS
            {
                DT cell_dot = 0;
                for (size_t k = 0; k < other._col_num; ++k)
                { // sum over shared dimension
                    cell_dot += other.at(r, k) * this->at(k, c);
                }
                new_mat._data.push_back(cell_dot);
            }
        }

        return new_mat;
    }
    matrix matmul_multithreaded(const matrix &other) const // left is = OTHER -> l_c = r_r
    {
        if (other._col_num != this->_row_num)
        {
            throw std::invalid_argument("Calling Other x Curr => Other Column Num != This Row Num");
        }

        matrix<DT> new_mat(other._row_num, _col_num, false);

        auto tile = [&](size_t r_t, size_t r_b)
        {
            for (size_t r = r_t; r < r_b; ++r) // LEFT MATRIX IS = OTHER
            {
                for (size_t c = 0; c < new_mat._col_num; ++c) // RIGHT MATRIX IS = THIS
                {
                    DT cell_dot = 0;
                    for (size_t k = 0; k < other._col_num; ++k)
                    { // sum over shared dimension
                        cell_dot += other.at(r, k) * this->at(k, c);
                    }
                    new_mat._data[r * new_mat._col_num + c] = cell_dot;
                }
            }
        };

        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        size_t rows_per_thread = other._row_num / num_threads;
        size_t r_start = 0;
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (size_t i = 0; i < num_threads; ++i)
        {
            size_t r_end = (i == num_threads - 1) ? other._row_num : r_start + rows_per_thread;
            threads.emplace_back(tile, r_start, r_end);
            r_start = r_end;
        }

        return new_mat;
    }

    matrix apply(std::function<DT(DT)> func) const
    {
        matrix<DT> result(this->_row_num, this->_col_num, false);
        _parallel_operator_overload([&](size_t r_start, size_t r_end)
                                    {
            for (size_t r = r_start; r < r_end; ++r) {
                for (size_t c{0}; c < this->_col_num; ++c) {
                    result.at(r, c) = func(this->at(r, c));
                }
            } }, this->_row_num);

        return result;
    }

    void apply(std::function<DT(DT)> func)
    {
        _parallel_operator_overload([&](size_t r_start, size_t r_end)
                                    {
            for (size_t r = r_start; r < r_end; ++r) {
                for (size_t c{0}; c < this->_col_num; ++c) {
                    this->at(r, c) = func(this->at(r, c));
                }
            } }, this->_row_num);
    }

    matrix<DT> mean(const size_t &axis) const
    {
        if (axis == 1) // row wise
        {
            matrix<DT> result(_row_num, 1, false); // one shared result matrix

            _parallel_operator_overload([&](size_t r_start, size_t r_end)
                                        {
            for (size_t r = r_start; r < r_end; ++r)
            {
                DT sum = 0;
                for (size_t c = 0; c < _col_num; ++c)
                    sum += at(r, c);
                result.at(r, 0) = sum / _col_num;  
            } }, _row_num);

            return result;
        }
        else if (axis == 0) // col-wise
        {
            matrix<DT> transposed = this->T(); // (col_num x row_num)
            return transposed.mean(1);         // row-wise on transposed → (col_num x 1)
        }
        else
        {
            throw std::invalid_argument("Axis can either be 1(row) or 0(col) in the form of an integer");
        }
    }

    void operator-=(const matrix &other)
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) -= other.at(r, c);
                }
            } }, this->_row_num);
    }

    matrix operator-(const matrix &other) const
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) - other.at(r, c);
                }
            } }, this->_row_num);

        return new_mat;
    }

    void operator+=(const matrix &other)
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) += other.at(r, c);
                }
            } }, this->_row_num);
    }

    matrix operator+(const matrix &other) const
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) + other.at(r, c);
                }
            } }, this->_row_num);

        return new_mat;
    }

    void operator*=(const matrix &other)
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) *= other.at(r, c);
                }
            } }, this->_row_num);
    }

    matrix operator*(const matrix &other) const
    {
        if (this->_row_num != other._row_num || this->_col_num != other._col_num)
        {
            throw std::invalid_argument("Element Wise Subtraction requires equivalent matrices");
        }

        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) * other.at(r, c);
                }
            } }, this->_row_num);

        return new_mat;
    }

    void operator-=(const DT &value)
    {
        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) -= value;
                }
            } }, this->_row_num);
    }

    matrix operator-(const DT &value) const
    {
        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) - value;
                }
            } }, this->_row_num);

        return new_mat;
    }

    void operator+=(const DT &value)
    {

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) += value;
                }
            } }, this->_row_num);
    }

    matrix operator+(const DT &value) const
    {

        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) + value;
                }
            } }, this->_row_num);

        return new_mat;
    }

    void operator*=(const DT &value)
    {

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    this->at(r, c) *= value;
                }
            } }, this->_row_num);
    }

    matrix operator*(const DT &value) const
    {

        matrix<DT> new_mat(this->_row_num, this->_col_num, false);

        _parallel_operator_overload([&](size_t row_start, size_t row_end)
                                    {
            for (size_t r = row_start; r < row_end; ++r)
            {
                for (size_t c{0}; c < this->_col_num; ++c)
                {
                    new_mat.at(r, c) = this->at(r, c) * value;
                }
            } }, this->_row_num);

        return new_mat;
    }

    void print() const
    {
        for (size_t r = 0; r < _row_num; ++r)
        {
            for (size_t c = 0; c < _col_num; ++c)
            {
                std::cout << at(r, c) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "--------\n";
    }
};