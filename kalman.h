#pragma once
#include "matrix.h"

class Kalman
{
private:
    matrix<double> x;
    matrix<double> P;
    matrix<double> F;
    matrix<double> H;
    matrix<double> Q;
    matrix<double> R;
    matrix<double> I;

public:
    Kalman(double x0, double y0, double dt)
    {
        x = matrix<double>(4, 1, false);
        x.at(0, 0) = x0;
        x.at(1, 0) = y0;
        x.at(2, 0) = 0;
        x.at(3, 0) = 0;

        P = matrix<double>(4, 4, false);
        P.at(0, 0) = 1;
        P.at(1, 1) = 1;
        P.at(2, 2) = 1;
        P.at(3, 3) = 1;

        F = matrix<double>(4, 4, false);
        F.at(0, 0) = 1;
        F.at(0, 2) = dt;
        F.at(1, 1) = 1;
        F.at(1, 3) = dt;
        F.at(2, 2) = 1;
        F.at(3, 3) = 1;

        H = matrix<double>(2, 4, false);
        H.at(0, 0) = 1;
        H.at(1, 1) = 1;

        Q = matrix<double>(4, 4, false);
        Q.at(0, 0) = 0.01;
        Q.at(1, 1) = 0.01;
        Q.at(2, 2) = 0.1;
        Q.at(3, 3) = 0.1;

        R = matrix<double>(2, 2, false);
        R.at(0, 0) = 0.5;
        R.at(1, 1) = 0.5;

        I = matrix<double>(4, 4, false);
        I.at(0, 0) = 1;
        I.at(1, 1) = 1;
        I.at(2, 2) = 1;
        I.at(3, 3) = 1;
    }

    void predict()
    {
        matrix<double> Ft = F.T();
        x = F.matmul(x);
        P = F.matmul(P).matmul(Ft) + Q;
    }

    void update(double mx, double my)
    {
        matrix<double> z(2, 1, false);
        z.at(0, 0) = mx;
        z.at(1, 0) = my;

        matrix<double> Ht = H.T();
        matrix<double> S = H.matmul(P).matmul(Ht) + R;
        matrix<double> Sinv = S.inverse();
        matrix<double> K = P.matmul(Ht).matmul(Sinv);

        matrix<double> y_err = z - H.matmul(x);
        x = x + K.matmul(y_err);
        P = (I - K.matmul(H)).matmul(P);
    }

    double get_x() const { return x.at(0, 0); }
    double get_y() const { return x.at(1, 0); }
    double get_vx() const { return x.at(2, 0); }
    double get_vy() const { return x.at(3, 0); }
};