#include "input.h"

void debug_print_input(layer input, const char *log_name);
void debug_print_weight(float *weight, int filters, int i_channel, int f_size, const char *log_name);
void debug_print_bias(float *bias, int channel, const char *log_name);