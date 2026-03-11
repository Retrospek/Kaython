#include <iostream>
#include <chrono>
#include <string>
#include "matrix.h"

// ─── helpers ────────────────────────────────────────────────────────────────

struct Result
{
    std::string name;
    long long ns;
};

template <typename Func>
Result time_it(const std::string &label, Func &&fn)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto stop = std::chrono::high_resolution_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return {label, ns};
}

void print_result(const Result &r)
{
    double ms = r.ns / 1e6;
    std::cout << "  [" << r.name << "] " << r.ns << " ns  (" << ms << " ms)\n";
}

void section(const std::string &title)
{
    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "══════════════════════════════════════════\n";
}

// ─── main ───────────────────────────────────────────────────────────────────

int main()
{
    constexpr size_t LARGE = 1000;
    constexpr size_t SMALL = 4;

    // ── 1. Constructors ──────────────────────────────────────────────────────
    section("Constructors");

    matrix<double> empty;
    std::cout << "  Default ctor: " << empty.get_row_num() << "x" << empty.get_col_num() << "\n";

    matrix<double> A(LARGE, LARGE, true);
    matrix<double> B(LARGE, LARGE, true);

    auto r_copy = time_it("Copy ctor (1000x1000)", [&]
                          { matrix<double> tmp(A); (void)tmp; });
    print_result(r_copy);

    auto r_move = time_it("Move ctor (1000x1000)", [&]
                          {
        matrix<double> src(A);
        matrix<double> dst(std::move(src));
        (void)dst; });
    print_result(r_move);

    auto r_copy_assign = time_it("Copy assignment (1000x1000)", [&]
                                 {
        matrix<double> tmp;
        tmp = A;
        (void)tmp; });
    print_result(r_copy_assign);

    auto r_move_assign = time_it("Move assignment (1000x1000)", [&]
                                 {
        matrix<double> src(A);
        matrix<double> dst;
        dst = std::move(src);
        (void)dst; });
    print_result(r_move_assign);

    // ── 2. push_row / at ─────────────────────────────────────────────────────
    section("push_row  &  at");

    {
        matrix<double> m;
        auto r = time_it("push_row x1000 (1000-wide)", [&]
                         {
            std::vector<double> row(1000, 1.5);
            for (int i = 0; i < 1000; ++i)
                m.push_row(row); });
        print_result(r);
        std::cout << "  Final size: " << m.get_row_num() << "x" << m.get_col_num() << "\n";
        std::cout << "  at(0,0)=" << m.at(0, 0) << "  at(999,999)=" << m.at(999, 999) << "\n";
    }

    // ── 3. Transpose ─────────────────────────────────────────────────────────
    section("Transpose");

    {
        matrix<double> s(SMALL, SMALL + 2, false);
        for (size_t r = 0; r < SMALL; ++r)
            for (size_t c = 0; c < SMALL + 2; ++c)
                s.at(r, c) = static_cast<double>(r * 10 + c);

        std::cout << "  Original (" << s.get_row_num() << "x" << s.get_col_num() << "):\n";
        s.print();

        const auto &s_ref = s;
        matrix<double> st = s_ref.T();
        std::cout << "  T() copy (" << st.get_row_num() << "x" << st.get_col_num() << "):\n";
        st.print();

        s.T();
        std::cout << "  T() in-place (" << s.get_row_num() << "x" << s.get_col_num() << "):\n";
        s.print();
    }

    {
        matrix<double> tmp(A);
        auto r1 = time_it("T() copy     (1000x1000)", [&]
                          {const auto& ref = tmp; auto x = ref.T(); (void)x; });
        auto r2 = time_it("T() in-place (1000x1000)", [&]
                          { tmp.T(); });
        print_result(r1);
        print_result(r2);
    }

    // ── 4. matmul ────────────────────────────────────────────────────────────
    section("matmul  (1000x1000 * 1000x1000)");

    matrix<double> C_single, C_multi;
    auto r_mm = time_it("matmul single-thread", [&]
                        { C_single = B.matmul(A); });
    auto r_mmt = time_it("matmul multithreaded", [&]
                         { C_multi = B.matmul_multithreaded(A); });
    print_result(r_mm);
    print_result(r_mmt);
    std::cout << "  Speedup: " << static_cast<double>(r_mm.ns) / r_mmt.ns << "x\n";

    // ── 5. Element-wise matrix operators ─────────────────────────────────────
    section("Element-wise matrix operators (1000x1000)");

    {
        matrix<double> a(A), b(B);
        auto r1 = time_it("operator+  (copy)", [&]
                          { auto x = a + b; (void)x; });
        auto r2 = time_it("operator-  (copy)", [&]
                          { auto x = a - b; (void)x; });
        auto r3 = time_it("operator*  (copy)", [&]
                          { auto x = a * b; (void)x; });
        auto r4 = time_it("operator+= (in-place)", [&]
                          { a += b; });
        auto r5 = time_it("operator-= (in-place)", [&]
                          { a -= b; });
        auto r6 = time_it("operator*= (in-place)", [&]
                          { a *= b; });
        for (auto &r : {r1, r2, r3, r4, r5, r6})
            print_result(r);
    }

    {
        matrix<double> p(SMALL, SMALL, false), q(SMALL, SMALL, false);
        for (size_t i = 0; i < SMALL; ++i)
            for (size_t j = 0; j < SMALL; ++j)
            {
                p.at(i, j) = 2.0;
                q.at(i, j) = 3.0;
            }

        std::cout << "\n  Spot-check (4x4, all-2 + all-3 = all-5):\n";
        (p + q).print();
    }

    // ── 6. Scalar operators ──────────────────────────────────────────────────
    section("Scalar operators (1000x1000, scalar=2.5)");

    {
        matrix<double> a(A);
        const double sc = 2.5;
        auto r1 = time_it("operator+  scalar (copy)", [&]
                          { auto x = a + sc; (void)x; });
        auto r2 = time_it("operator-  scalar (copy)", [&]
                          { auto x = a - sc; (void)x; });
        auto r3 = time_it("operator*  scalar (copy)", [&]
                          { auto x = a * sc; (void)x; });
        auto r4 = time_it("operator+= scalar (in-place)", [&]
                          { a += sc; });
        auto r5 = time_it("operator-= scalar (in-place)", [&]
                          { a -= sc; });
        auto r6 = time_it("operator*= scalar (in-place)", [&]
                          { a *= sc; });
        for (auto &r : {r1, r2, r3, r4, r5, r6})
            print_result(r);
    }

    {
        matrix<double> p(SMALL, SMALL, false);
        for (size_t i = 0; i < SMALL; ++i)
            for (size_t j = 0; j < SMALL; ++j)
                p.at(i, j) = 4.0;
        std::cout << "\n  Spot-check (4x4, all-4 * 3.0 = all-12):\n";
        (p * 3.0).print();
    }

    // ── 7. Mean ──────────────────────────────────────────────────────────────
    section("Mean (1000x1000)");

    {
        size_t axis1 = 1;
        size_t axis0 = 0;

        matrix<double> row_result, col_result;
        auto r1 = time_it("mean(axis=1) row-wise  → (1000x1)", [&]
                          { row_result = A.mean(axis1); });
        auto r2 = time_it("mean(axis=0) col-wise  → (1x1000)", [&]
                          { col_result = A.mean(axis0); });
        print_result(r1);
        print_result(r2);

        std::cout << "  row_result size: " << row_result.get_row_num() << "x" << row_result.get_col_num() << "\n";
        std::cout << "  col_result size: " << col_result.get_row_num() << "x" << col_result.get_col_num() << "\n";
    }

    // correctness spot-check
    {
        matrix<double> p(SMALL, SMALL, false);
        for (size_t i = 0; i < SMALL; ++i)
            for (size_t j = 0; j < SMALL; ++j)
                p.at(i, j) = static_cast<double>((i + 1) * 10); // rows: 10, 20, 30, 40

        size_t axis1 = 1;
        size_t axis0 = 0;

        std::cout << "\n  Spot-check input (rows are 10, 20, 30, 40):\n";
        p.print();

        std::cout << "  mean(axis=1) — each row mean should be 10, 20, 30, 40:\n";
        p.mean(axis1).print();

        std::cout << "  mean(axis=0) — each col mean should be 25:\n";
        p.mean(axis0).print();
    }

    // ── 8. Done ──────────────────────────────────────────────────────────────
    section("Done — all tests complete");
    std::cout << "\n";
    return 0;
}