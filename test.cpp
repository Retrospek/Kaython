#include <utility>
#include <vector>
#include <iostream>
#include "ball.h"

// OpenGL rendering
#include <GLFW/glfw3.h>

int main()
{
    Ball ball(10, 1, 1, 10, 100);

    // 1. Collect positions
    size_t N = 100;
    std::vector<std::pair<long double, long double>> positions;
    positions.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        positions.push_back(ball._curr_pos);
        ball.move();
    }

    // 2. Find bounding box for normalization
    long double xmin = positions[0].first, xmax = positions[0].first;
    long double ymin = positions[0].second, ymax = positions[0].second;
    for (auto &[x, y] : positions)
    {
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
        ymin = std::min(ymin, y);
        ymax = std::max(ymax, y);
    }
    // Avoid divide-by-zero if ball doesn't move
    long double xrange = (xmax - xmin) < 1e-9 ? 1.0 : (xmax - xmin);
    long double yrange = (ymax - ymin) < 1e-9 ? 1.0 : (ymax - ymin);

    // 3. Init GLFW + OpenGL window
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(800, 800, "Ball Path", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        // Draw path as a line strip
        glColor3f(0.2f, 0.8f, 1.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for (auto &[x, y] : positions)
        {
            // Normalize to [-0.9, 0.9]
            float nx = (float)((x - xmin) / xrange) * 1.8f - 0.9f;
            float ny = (float)((y - ymin) / yrange) * 1.8f - 0.9f;
            glVertex2f(nx, ny);
        }
        glEnd();

        // Draw start (green dot) and end (red dot)
        auto drawDot = [&](long double x, long double y, float r, float g, float b)
        {
            float nx = (float)((x - xmin) / xrange) * 1.8f - 0.9f;
            float ny = (float)((y - ymin) / yrange) * 1.8f - 0.9f;
            glColor3f(r, g, b);
            glPointSize(10.0f);
            glBegin(GL_POINTS);
            glVertex2f(nx, ny);
            glEnd();
        };
        drawDot(positions.front().first, positions.front().second, 0.0f, 1.0f, 0.0f); // green = start
        drawDot(positions.back().first, positions.back().second, 1.0f, 0.2f, 0.2f);   // red   = end

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}