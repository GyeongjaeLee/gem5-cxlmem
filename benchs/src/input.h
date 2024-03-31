#ifndef __INPUT_H_
#define __INPUT_H_

struct layer {
	int width;
    int channel;
	float *data;
};

typedef struct layer layer;
void load_file(float *w_input, const char *file_name, int f_size);
void print_result(float prob, int index);
void softmax(float *input, int items);

#endif