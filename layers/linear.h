#include "matrix.h"
#include <cmath>

template <
    typename TW, // type used for X
    typename TB, // type used for y
    typename TH  // type used for input history
    >
class linear
{
private:
    matrix<TW> weights;
    matrix<TB> biases;

    matrix<TH> history;

public:
};