#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "linear_regression.h"

// ─── pretty print ─────────────────────────────────────────────

void header(const std::string &title)
{
    std::cout << "\n\033[1;36m━━━ " << title << " ━━━\033[0m\n";
}

void kv(const std::string &k, float v)
{
    std::cout << "  \033[90m" << std::setw(18) << std::left << k
              << "\033[1;97m" << std::fixed << std::setprecision(6) << v << "\033[0m\n";
}

void loss_curve(const std::vector<float> &losses, int samples = 8)
{
    float lo = losses.back(), hi = losses.front();
    size_t stride = losses.size() / samples;

    std::cout << "\n  Loss trend:\n";
    for (int i = 0; i < samples; ++i)
    {
        size_t idx = i * stride;
        float l = losses[idx];
        int bar = (int)(20.0f * (l - lo) / (hi - lo + 1e-9f));

        std::cout << "  [" << std::setw(4) << idx << "] ";
        std::cout << std::string(bar, '|') << "\n";
    }
    std::cout << "  [" << losses.size() - 1 << "] final\n";
}

// ─── data ───────────────────────────────────────────────────

void make_data(matrix<float> &X, matrix<float> &y,
               size_t N,
               const std::vector<float> &w,
               float b)
{
    size_t F = w.size();

    for (size_t i = 0; i < N; ++i)
    {
        float yi = b;

        for (size_t f = 0; f < F; ++f)
        {
            float x = (i + f + 1) * 0.1f;
            X.at(i, f) = x;
            yi += w[f] * x;
        }

        y.at(i, 0) = yi;
    }
}

// ─── experiment runner ───────────────────────────────────────

void run_experiment(
    size_t N,
    std::vector<float> true_w,
    float true_b,
    float LR,
    size_t iters,
    float l2 = 0.0f,
    float l1 = 0.0f)
{
    size_t F = true_w.size();

    header("Linear Regression Experiment");

    // build dataset
    matrix<float> X(N, F, false), y(N, 1, false);
    make_data(X, y, N, true_w, true_b);

    // model
    LinearRegression<float, float, float, float> model(F, LR, "mse", l2, l1);

    // train
    auto losses = model.fit(X, y, iters);

    // summary
    kv("N", (float)N);
    kv("F", (float)F);
    kv("LR", LR);
    kv("first loss", losses.front());
    kv("final loss", losses.back());

    loss_curve(losses);

    // quick sanity prediction
    matrix<float> x_test(1, F, false);
    for (size_t f = 0; f < F; ++f)
        x_test.at(0, f) = 5.0f;

    auto pred = model.predict(x_test);

    float expected = true_b;
    for (size_t f = 0; f < F; ++f)
        expected += true_w[f] * 5.0f;

    std::cout << "\n  Prediction check:\n";
    kv("pred(x=5,...)", pred.at(0, 0));
    kv("truth", expected);
}

// ─── main ───────────────────────────────────────────────────

int main()
{
    std::cout << "\n\033[1;97mLinear Regression Sandbox\033[0m\n";

    // 🔧 JUST CHANGE THESE ↓↓↓

    size_t N = 100;
    std::vector<float> weights = {4.0f}; // change feature count here
    float bias = 7.0f;

    float LR = 0.0025f;
    size_t iters = 5000;

    float l2 = 0.0f;
    float l1 = 0.0f;

    // 🔧 run
    run_experiment(N, weights, bias, LR, iters, l2, l1);

    return 0;
}