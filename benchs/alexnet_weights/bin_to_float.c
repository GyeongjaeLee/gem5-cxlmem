#include <stdio.h>
#include <stdlib.h>

int main() {
	
	int i;
	int channel, row, col;
	float *buf;
	const char *b_name[4] = {"bicycle227.bin", "cat227.bin", "cheetah227.bin", "fox227.bin"};
	const char *n_name[4] = {"bicycle227.txt", "cat227.txt", "cheetah227.txt", "fox227.txt"};
	FILE *fp_w, *fp_r;
	
	buf = (float *)malloc(sizeof(float) * 227 * 227 * 3);
	for(i=0; i<4; i++) {
		fp_r = fopen(b_name[i], "r");
		fp_w = fopen(n_name[i], "w");
		fread(buf, sizeof(float), 227*227*3, fp_r);
		for(channel=0; channel<3; channel++) {
			fprintf(fp_w, "channel #%2d\n", channel+1);
			for(row=0; row<227; row++) {
				fprintf(fp_w, "row #%4d\n", row+1);
				for(col=0; col<227; col++) {
					fprintf(fp_w, "%f  ", buf[channel*227*227 + row*227 + col]);
				}
				fprintf(fp_w, "\n");
			}
		}
		fclose(fp_r);
		fclose(fp_w);
	}
	free(buf);
	
	return 0;
}