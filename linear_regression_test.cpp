#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "linear_regression.h"

void print_section_header(const std::string &title)
{
    std::cout << "\n\033[1;36m━━━ " << title << " ━━━\033[0m\n";
}

void print_key_value(const std::string &key, float value)
{
    std::cout << "  \033[90m" << std::setw(18) << std::left << key
              << "\033[1;97m" << std::fixed << std::setprecision(6) << value << "\033[0m\n";
}

void visualize_loss_progression(const std::vector<float> &losses, int samples = 8)
{
    float minimum_loss = losses.back();
    float maximum_loss = losses.front();
    size_t step_size = losses.size() / samples;
    
    std::cout << "\n  Loss trend:\n";
    for (int i = 0; i < samples; ++i)
    {
        size_t index = i * step_size;
        float current_loss = losses[index];
        int bar_length = (int)(20.0f * (current_loss - minimum_loss) / (maximum_loss - minimum_loss + 1e-9f));
        
        std::cout << "  [" << std::setw(4) << index << "] ";
        std::cout << std::string(bar_length, '|') << "\n";
    }
    std::cout << "  [" << losses.size() - 1 << "] final\n";
}

void generate_synthetic_data(matrix<float> &features, matrix<float> &targets,
                             size_t num_samples,
                             const std::vector<float> &true_weights,
                             float true_bias)
{
    size_t num_features = true_weights.size();
    
    for (size_t sample = 0; sample < num_samples; ++sample)
    {
        float target_value = true_bias;
        
        for (size_t feature = 0; feature < num_features; ++feature)
        {
            float feature_value = (sample + feature + 1) * 0.1f;
            features.at(sample, feature) = feature_value;
            target_value += true_weights[feature] * feature_value;
        }
        
        targets.at(sample, 0) = target_value;
    }
}

void run_linear_regression_experiment(
    size_t num_samples,
    std::vector<float> true_weights,
    float true_bias,
    float learning_rate,
    size_t iterations,
    float l2_regularization = 0.0f,
    float l1_regularization = 0.0f)
{
    size_t num_features = true_weights.size();
    
    print_section_header("Linear Regression Experiment");
    
    matrix<float> features(num_samples, num_features, false);
    matrix<float> targets(num_samples, 1, false);
    generate_synthetic_data(features, targets, num_samples, true_weights, true_bias);
    
    LinearRegression<float, float, float, float> model(
        num_features, learning_rate, "mse", l2_regularization, l1_regularization);
    
    auto training_losses = model.fit(features, targets, iterations);
    
    print_key_value("N", (float)num_samples);
    print_key_value("F", (float)num_features);
    print_key_value("LR", learning_rate);
    print_key_value("first loss", training_losses.front());
    print_key_value("final loss", training_losses.back());
    
    visualize_loss_progression(training_losses);
    
    matrix<float> test_input(1, num_features, false);
    for (size_t feature = 0; feature < num_features; ++feature)
        test_input.at(0, feature) = 5.0f;
    
    auto prediction = model.predict(test_input);
    
    float expected_output = true_bias;
    for (size_t feature = 0; feature < num_features; ++feature)
        expected_output += true_weights[feature] * 5.0f;
    
    std::cout << "\n  Prediction check:\n";
    print_key_value("pred(x=5,...)", prediction.at(0, 0));
    print_key_value("truth", expected_output);
}

int main()
{
    std::cout << "\n\033[1;97mLinear Regression Sandbox\033[0m\n";
    
    size_t num_samples = 100;
    std::vector<float> weights = {4.0f};
    float bias = 7.0f;
    float learning_rate = 0.0025f;
    size_t iterations = 5000;
    float l2_regularization = 0.0f;
    float l1_regularization = 0.0f;
    
    run_linear_regression_experiment(
        num_samples, weights, bias, learning_rate, iterations, 
        l2_regularization, l1_regularization);
    
    return 0;
}
