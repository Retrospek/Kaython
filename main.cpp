#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <random>
#include <iomanip>
#include <sstream>
#include <vector>
#include "matrix.h"
#include "kalman.h"

// ─── Ball ───────────────────────────────────────────────────────────────────

struct Ball
{
    double x, y;
    double vx, vy;
    double ax, ay;

    Ball(double x0, double y0, double vx0, double vy0)
        : x(x0), y(y0), vx(vx0), vy(vy0), ax(0.0), ay(-9.8) {}

    void step(double dt)
    {
        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;
    }

    bool in_flight() const { return y >= 0.0; }
};

// ─── Noise ──────────────────────────────────────────────────────────────────

struct NoisySensor
{
    std::mt19937 rng{123};
    std::normal_distribution<double> noise;

    NoisySensor(double stddev) : noise(0.0, stddev) {}

    std::pair<double, double> observe(const Ball &b)
    {
        return {b.x + noise(rng), b.y + noise(rng)};
    }
};

// ─── Point ──────────────────────────────────────────────────────────────────

struct Point
{
    double x, y;
};

// ─── Main ───────────────────────────────────────────────────────────────────

int main()
{
    constexpr double dt = 0.1;
    constexpr double noise = 2.5;

    auto fmt = [](double a, double b)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << "(" << a << ", " << b << ")";
        return ss.str();
    };

    // ─── Simulate & store ───────────────────────────────────────────────────

    std::vector<Point> true_pts, noisy_pts, kalman_pts;

    Ball ball(0.0, 0.0, 20.0, 40.0);
    NoisySensor sensor(noise);
    Kalman kf(ball.x, ball.y, dt);

    while (ball.in_flight())
    {
        kf.predict();
        auto [mx, my] = sensor.observe(ball);
        kf.update(mx, my);

        true_pts.push_back({ball.x, ball.y});
        noisy_pts.push_back({mx, my});
        kalman_pts.push_back({kf.get_x(), kf.get_y()});

        ball.step(dt);
    }

    // ─── Table ──────────────────────────────────────────────────────────────

    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "  Kalman Filter -- 2D Projectile Tracking\n";
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "  Launch: vx=20 m/s  vy=40 m/s  gravity=-9.8 m/s²  noise=" << noise << "m\n\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left
              << std::setw(8) << "  t(s)"
              << std::setw(26) << "  True (x,y)"
              << std::setw(26) << "  Noisy (x,y)"
              << std::setw(26) << "  Kalman (x,y)"
              << std::setw(26) << "  Noisy Err (x,y)"
              << std::setw(26) << "  Kalman Err (x,y)"
              << "\n";
    std::cout << std::string(138, '-') << "\n";

    for (size_t i = 0; i < true_pts.size(); ++i)
    {
        double t = i * dt;
        auto &tp = true_pts[i];
        auto &np = noisy_pts[i];
        auto &kp = kalman_pts[i];

        double noisy_err_x = np.x - tp.x;
        double noisy_err_y = np.y - tp.y;
        double kalman_err_x = kp.x - tp.x;
        double kalman_err_y = kp.y - tp.y;

        std::ostringstream t_ss;
        t_ss << std::fixed << std::setprecision(2) << t;

        std::cout << std::left
                  << std::setw(8) << ("  " + t_ss.str())
                  << std::setw(26) << ("  " + fmt(tp.x, tp.y))
                  << std::setw(26) << ("  " + fmt(np.x, np.y))
                  << std::setw(26) << ("  " + fmt(kp.x, kp.y))
                  << std::setw(26) << ("  " + fmt(noisy_err_x, noisy_err_y))
                  << std::setw(26) << ("  " + fmt(kalman_err_x, kalman_err_y))
                  << "\n";
    }

    // ─── Landing prediction ─────────────────────────────────────────────────

    double kf_vx = kf.get_vx();
    double kf_vy = kf.get_vy();
    double kf_x = kf.get_x();
    double kf_y = kf.get_y();

    double disc = kf_vy * kf_vy + 2 * 9.8 * kf_y;
    double t_land = (kf_vy + std::sqrt(disc)) / 9.8;
    double predicted_landing = kf_x + kf_vx * t_land;
    double true_landing = ball.x - ball.vx * dt;

    std::cout << "\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "  Predicted landing x : " << predicted_landing << " m\n";
    std::cout << "  Approx true landing : " << true_landing << " m\n";
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n";

    // ─── ASCII Plot ─────────────────────────────────────────────────────────

    constexpr int W = 160;
    constexpr int H = 50;

    double x_min = 0, x_max = true_pts.back().x;
    double y_min = 0, y_max = 0;
    for (auto &p : true_pts)
        y_max = std::max(y_max, p.y);

    auto to_col = [&](double x)
    { return (int)((x - x_min) / (x_max - x_min) * (W - 1)); };
    auto to_row = [&](double y)
    { return (int)((1.0 - (y - y_min) / (y_max - y_min)) * (H - 1)); };

    std::vector<std::string> grid(H, std::string(W, ' '));

    // true lowest priority
    for (auto &p : true_pts)
    {
        int c = to_col(p.x), r = to_row(p.y);
        if (r >= 0 && r < H && c >= 0 && c < W)
            grid[r][c] = '*';
    }
    // noisy medium priority
    for (auto &p : noisy_pts)
    {
        int c = to_col(p.x), r = to_row(p.y);
        if (r >= 0 && r < H && c >= 0 && c < W)
            grid[r][c] = '.';
    }
    // kalman highest priority
    for (auto &p : kalman_pts)
    {
        int c = to_col(p.x), r = to_row(p.y);
        if (r >= 0 && r < H && c >= 0 && c < W)
            grid[r][c] = '#';
    }

    std::cout << "  ASCII Trajectory Plot\n";
    std::cout << "  * = True   . = Noisy   # = Kalman\n\n";
    for (auto &row : grid)
        std::cout << "  |" << row << "|\n";
    std::cout << "  +" << std::string(W, '-') << "+\n\n";

    return 0;
}