#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void load_file(float **w_input, const char *file_name, int f_length) {
    int i;
    FILE *fp;
    if(*w_input == NULL)
        *w_input = (float *)malloc(f_length*sizeof(float));
    else
        *w_input = (float *)realloc(*w_input, f_length*sizeof(float));
	fp = fopen(file_name, "r");
	fread(*w_input, sizeof(float), f_length, fp);
	fclose(fp);
}

void print_result(float prob, int index) {	
    int i;
	FILE *fp = fopen("label_name.txt","r");
	char word[200];

	for(i=0; i<index+1; i++)
		fgets(word, sizeof(word), fp);
	word[strlen(word)-1] = '\0';

	printf("%f%% - \"%s\"\n", prob, word);
	fclose(fp);	
}

void softmax(float *input, int items) {
    
    int i;
    int top[5] = {0, 0, 0, 0, 0};
    float sum = 0.0f;
    float softmax[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for(i=0; i<items; i++) {
        input[i] = expf(input[i]);
        sum += input[i];
    }
    
    /* Showing percentage */
    for(i=0; i<items; i++)
        input[i] = 100.0f * input[i] / sum;
    
    /* choosing top-5 */
    for (i=0; i<items; i++) {
		if(input[i] > softmax[0]) {
			softmax[4] = softmax[3]; top[4] = top[3];
			softmax[3] = softmax[2]; top[3] = top[2];
			softmax[2] = softmax[1]; top[2] = top[1];
			softmax[1] = softmax[0]; top[1] = top[0];
			softmax[0] = input[i]; top[0] = i;
		}
		else if(input[i] > softmax[1]) {
			softmax[4] = softmax[3]; top[4] = top[3];
			softmax[3] = softmax[2]; top[3] = top[2];
			softmax[2] = softmax[1]; top[2] = top[1];
			softmax[1] = input[i]; top[1] = i;
		}
		else if (input[i] > softmax[2]) {
			softmax[4] = softmax[3]; top[4] = top[3];
			softmax[3] = softmax[2]; top[3] = top[2];
			softmax[2] = input[i]; top[2] = i;
		}
		else if (input[i] > softmax[3]) {
			softmax[4] = softmax[3]; top[4] = top[3];
			softmax[3] = input[i]; top[3] = i;
		}
		else if (input[i] > softmax[4]) {
			softmax[4] = input[i]; top[4] = i;
		}
	}
    for(i=0; i<5; i++)
        print_result(softmax[i],top[i]);
}




