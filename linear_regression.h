#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "matrix.h"

template <
    typename TX,
    typename TY,
    typename TW,
    typename TL>
class LinearRegression
{
private:
    matrix<TW> weights;
    TW bias;
    float LR;
    std::string criterion_selection;
    float l1_lambda;
    float l2_lambda;

public:
    ~LinearRegression() = default;

    LinearRegression(const size_t &num_features, const float &LR,
                     const std::string &criterion_selection, const float &l2_lambda, const float &l1_lambda)
        : weights(1, num_features, true), bias(0), // no `new`, weights is a value not a pointer
          LR(LR), criterion_selection(criterion_selection), l1_lambda(l1_lambda), l2_lambda(l2_lambda)
    {
    }

    std::vector<TL> fit(const matrix<TX> &X, const matrix<TY> &y, const size_t &num_iters) // refs go after type
    {
        std::vector<TL> losses;
        losses.reserve(num_iters);

        for (size_t i{0}; i < num_iters; ++i)
        {
            matrix<TY> predictions = this->predict(X);
            TL loss = this->MSE_Loss(predictions, y);
            losses.push_back(loss);
            std::cout << "Iteration: " << i << ", Loss: " << loss << "\n";

            std::pair<matrix<TW>, TW> gradient = this->MSE_Grad(predictions, y, X); // was missing X arg

            this->weights -= gradient.first * LR;
            this->bias -= gradient.second * LR;
        }

        return losses;
    }

    matrix<TY> predict(const matrix<TX> &X)
    {
        matrix<TW> wT = weights;
        wT.T();
        matrix<TY> predictions = wT.matmul_multithreaded(X);
        predictions += this->bias;
        return predictions;
    }

    TL MSE_Loss(const matrix<TY> &predictions, const matrix<TY> &truths)
    {
        matrix<TY> losses = predictions - truths;
        losses = losses * losses;
        TL mse = losses.mean(1).at(0, 0);

        return mse;
    }

    std::pair<matrix<TW>, TW> MSE_Grad(const matrix<TY> &predictions, const matrix<TY> &truths, const matrix<TX> &X)
    {
        size_t N = X.get_row_num();
        matrix<TY> residual = predictions - truths; // (N, 1)

        // dL/dw = (2/N) * X.T @ residual → (F,N) @ (N,1) = (F,1) ... then T to get (1,F)
        matrix<TX> XT = X;
        XT.T();                                                     // (F, N)
        matrix<TW> weight_grad = residual.matmul_multithreaded(XT); // XT @ residual = (F,N)@...
        // matmul(other) = other @ this, so residual.matmul(XT) = XT @ residual = (F,N)@(N,1) = (F,1)
        weight_grad.T(); // (1, F) to match weights shape
        weight_grad *= TW(2) / static_cast<TW>(N);

        TW bias_grad = TW(0);
        for (size_t r = 0; r < N; ++r)
            bias_grad += residual.at(r, 0);
        bias_grad *= TW(2) / static_cast<TW>(N);

        return {weight_grad, bias_grad};
    }
};