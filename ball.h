#include <utility>
#include <vector>
#include <iostream>

struct Ball
{
    long double _radius;
    std::pair<long double, long double> _curr_pos;
    std::pair<long double, long double> _curr_velocity;
    long double gravity = -9.8;

    Ball(long double r, long double x, long double y, long double dx, long double dy)
        : _radius(r), _curr_pos({x, y}), _curr_velocity({dx, dy})
    {
    }

    void move()
    {
        _curr_pos.first += _curr_velocity.first;
        _curr_pos.second += _curr_velocity.second + gravity;

        _curr_velocity.second += gravity;
    }
};