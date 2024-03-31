#include "input.h"
layer zero_pad(layer input, int pad);
layer conv_layer(layer input, int filter, int f_size, int stride, int pad, float *weight, float *bias);
