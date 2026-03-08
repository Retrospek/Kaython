#include <iostream>
#include <string>
#include <vector>
#include "matrix.h"
#include <string>

template <
    typename TX, // type used for X
    typename TY, // type used for y
    typename TW, // type used for weights/bias
    typename TL  // type used for losses and gradients
    >
class LinearRegression
{
private:
    matrix<TW> *weights; // define pointers for move semantics in the future with weight updates
    matrix<TW> *biases;
    float LR;
    std::string criterion_selection;
    float l1_lambda;
    float l2_lambda;

public:
    ~LinearRegression() default;

    LinearRegression(const size_t &num_features, const float &LR,
                     const &std::string criterion_selection, const float &l2_lambda, const float &l1_lambda)
        : weights(new matrix<TW>(1, num_features, true)), biases(new matrix<TW>(1, num_features, true)),
          LR(LR), criterion_selection(criterion_selection), l1_lambda(l1_lambda), l2_lambda(l2_lambda)
    {
    }

    const &std::vector<TL> fit(const &matrix<TX> X, const &matrix<TY> y, const &size_t num_iters)
    {
        std::vector<TL> losses;
        losses.reserve(num_iters);

        for (size_t i{0}; i < num_iters; ++i)
        {
            const TY predictions = this->predict(X);
            const TY loss = this->MSE_Loss(predictions, y);
            losses[i] = loss;

            weight_grad = this->MSE_Weight_Grad(predictions, y);
            bias_grad = this->MSE_Bias_Grad(predictions, y);
        }
    }

    const &TY predict(const &matrix<TX> X)
    {
    }

    const &TY MSE_Loss(const &<TY> predictions, const &<TY> truths)
    {
    }

    const &matrix<TW> MSE_Weight_Grad(const &<TY> predictions, const &<TY> truths)
    {
    }
    const &matrix<TW> MSE_Bias_Grad(const &<TY> predictions, const &<TY> truths)
    {
    }
};