#include "bench_lstm.h"

//@@@@@@@@@@@@@@@@@@@@@@ parameter definition @@@@@@@@@@@@@@@@@@@@@@@@@
#define NUM_PE          	8
#define INPUT_WIDTH			1
#define	INPUT_SEQUENCE		1
#define	BATCH_SIZE			16

#define INPUT_LENGTH		10
#define GENERATE_LENGTH  	200






//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ main @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

int main()
{
	int i;
	int batch;
	int sequence;

	layer *input = (layer *)malloc(sizeof(layer));
	layer *lstm_1 = (layer *)malloc(sizeof(layer));
	layer *lstm_2 = (layer *)malloc(sizeof(layer));
	layer *lstm_3 = (layer *)malloc(sizeof(layer));
	layer *fc_4 = (layer *)malloc(sizeof(layer));
	layer *softm = (layer *)malloc(sizeof(layer));
	layer *even = (layer *)malloc(sizeof(layer));

	input->batch = BATCH_SIZE;
	input->width = 1;
	input->sequence = INPUT_SEQUENCE;
	input->data  = (float **)malloc(BATCH_SIZE * sizeof(float *));

	lstm_1->batch = BATCH_SIZE;
	lstm_1->width = 50;
	lstm_1->sequence = INPUT_SEQUENCE;
	lstm_1->data = (float **)malloc(BATCH_SIZE * sizeof(float *));
	
	lstm_2->batch = BATCH_SIZE;
	lstm_2->width = 50;
	lstm_2->sequence = INPUT_SEQUENCE;
	lstm_2->data = (float **)malloc(BATCH_SIZE * sizeof(float *));


	lstm_3->batch = BATCH_SIZE;
	lstm_3->width = 50;
	lstm_3->sequence = INPUT_SEQUENCE;
	lstm_3->data = (float **)malloc(BATCH_SIZE * sizeof(float *));
	
	fc_4->batch = BATCH_SIZE;
	fc_4->width = 1;
	fc_4->sequence = INPUT_SEQUENCE;
	fc_4->data   = (float **)malloc(BATCH_SIZE * sizeof(float *));



	float *Uf_1 = NULL;    float *Uc_1 = NULL;    float *Ui_1 = NULL;    float *Uo_1 = NULL;
	float *Wf_1 = NULL;    float *Wc_1 = NULL;    float *Wi_1 = NULL;    float *Wo_1 = NULL;
	float *weight_fc = NULL, *bias_fc = NULL;	

	float **cell_memory_1 = (float **)malloc(BATCH_SIZE * sizeof(float *));
	float **cell_output_1 = (float **)malloc(BATCH_SIZE * sizeof(float *));
	
	float **cell_memory_2 = (float **)malloc(BATCH_SIZE * sizeof(float *));
	float **cell_output_2 = (float **)malloc(BATCH_SIZE * sizeof(float *));
	
	float **cell_memory_3 = (float **)malloc(BATCH_SIZE * sizeof(float *));
	float **cell_output_3 = (float **)malloc(BATCH_SIZE * sizeof(float *));



	for (batch = 0; batch < input->batch; batch++)
	{
		cell_memory_1[batch] = (float *)calloc(lstm_1->width, sizeof(float));
		cell_output_1[batch] = (float *)calloc(lstm_1->width, sizeof(float));   
		cell_memory_2[batch] = (float *)calloc(lstm_2->width, sizeof(float));
		cell_output_2[batch] = (float *)calloc(lstm_2->width, sizeof(float));   
		cell_memory_3[batch] = (float *)calloc(lstm_3->width, sizeof(float));
		cell_output_3[batch] = (float *)calloc(lstm_3->width, sizeof(float));   

		input->data[batch]  = (float *)malloc(sizeof(float)* input->width  * INPUT_SEQUENCE);
		lstm_1->data[batch] = (float *)malloc(sizeof(float)* lstm_1->width * INPUT_SEQUENCE);
		lstm_2->data[batch] = (float *)malloc(sizeof(float)* lstm_2->width * INPUT_SEQUENCE);
		lstm_3->data[batch] = (float *)malloc(sizeof(float)* lstm_3->width * INPUT_SEQUENCE);
		fc_4->data[batch]   = (float *)malloc(sizeof(float)* fc_4->width   * INPUT_SEQUENCE);

	}



	//################# load weights, biases, etc.. ###################

	printf("loading weights\n");

	load_file(&Uf_1, NAME_LSTM_1_UF, LENGTH_LSTM_1_UF);     
	load_file(&Uc_1, NAME_LSTM_1_UC, LENGTH_LSTM_1_UC);     
	load_file(&Ui_1, NAME_LSTM_1_UI, LENGTH_LSTM_1_UI);     
	load_file(&Uo_1, NAME_LSTM_1_UO, LENGTH_LSTM_1_UO);     

	load_file(&Wf_1, NAME_LSTM_1_WF, LENGTH_LSTM_1_WF);     
	load_file(&Wc_1, NAME_LSTM_1_WC, LENGTH_LSTM_1_WC);     
	load_file(&Wi_1, NAME_LSTM_1_WI, LENGTH_LSTM_1_WI);     
	load_file(&Wo_1, NAME_LSTM_1_WO, LENGTH_LSTM_1_WO);    

	load_file(&weight_fc, NAME_FC_W, LENGTH_FC_W);
	load_file(&bias_fc, NAME_FC_B, LENGTH_FC_B);

	printf("weights loading done\n\n");

	//############### sample input #####################
	
	
	

	float **stock_price = (float **)malloc(sizeof(float *) * input->batch);
	
	for(batch = 0; batch < input->batch; batch++)
		stock_price[batch] = (float *)malloc(sizeof(float) * (GENERATE_LENGTH + 1));





	//############### Generate #####################
	for(sequence = 0; sequence < GENERATE_LENGTH; sequence++)
	{

		printf("sequence : %d/%d\n", sequence+1, GENERATE_LENGTH);
		
		if(sequence < INPUT_LENGTH)
			generate_sample_input(input);
		
		
		// LSTM1
		lstm_layer(input, lstm_1, cell_memory_1, cell_output_1, Uf_1, Wf_1, Uc_1, Wc_1, Ui_1, Wi_1, Uo_1, Wo_1);
		printf("lstmlayer 1 done! ");
		// LSTM2
		lstm_layer(lstm_1, lstm_2, cell_memory_2, cell_output_2, Uf_1, Wf_1, Uc_1, Wc_1, Ui_1, Wi_1, Uo_1, Wo_1);
		printf("lstmlayer 2 done! ");
		// LSTM3
		lstm_layer(lstm_2, lstm_3, cell_memory_3, cell_output_3, Uf_1, Wf_1, Uc_1, Wc_1, Ui_1, Wi_1, Uo_1, Wo_1);
		printf("lstmlayer 3 done! ");


		if(sequence < INPUT_LENGTH)
		{
			for(batch = 0; batch < input->batch; batch++)
				stock_price[batch][sequence] = input->data[batch][0];
		}

		if(sequence >= INPUT_LENGTH - 1)
		{
			// FC4
			fc_layer(lstm_3, fc_4, weight_fc, bias_fc, 0);
			// SAVE_SYMBOL 
			for(batch = 0; batch < input->batch; batch++)
			{
				stock_price[batch][sequence + 1] = fc_4->data[batch][0];
				input->data[batch][0] = fc_4->data[batch][0];
			}
		}
	}


	// PRINT_SYMBOL
	print_symbol(stock_price);

}







//@@@@@@@@@@@@@@@@@@@@@@@@@@@@ function @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

void lstm_layer(layer *input, layer *output, float **cell_memory, float **cell_output, float *Uf, float *Wf, float *Uc, float *Wc, float *Ui, float *Wi, float *Uo, float *Wo)
{

	int batch;
	int sequence;
	int i;


	//######## start #########

	float **temp1 = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **temp2 = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **temp3 = (float **)malloc(sizeof(float *)* BATCH_SIZE);

	float **cell_temp1 = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **cell_temp2 = (float **)malloc(sizeof(float *)* BATCH_SIZE);

	float **forget_output = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **gate_output = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **input_output = (float **)malloc(sizeof(float *)* BATCH_SIZE);
	float **output_output = (float **)malloc(sizeof(float *)* BATCH_SIZE);


	for (batch = 0; batch < BATCH_SIZE; batch++)
	{
		temp1[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE*2);
		temp2[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE*2);
		temp3[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE*2);

		cell_temp1[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE*2);
		cell_temp2[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE*2);

		forget_output[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE);
		gate_output[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE);
		input_output[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE);
		output_output[batch] = (float *)malloc(sizeof(float)* output->width * INPUT_SEQUENCE);
	}



	for (sequence = 0; sequence < input->sequence; sequence++)
	{
		//forget
		forward(input->data, temp1, input->width, output->width, Uf, sequence, 0);

		forward(cell_output, temp2, input->width, output->width, Wf, sequence, 1);
		element_add(temp1, temp2, temp3, output->width, sequence);
		sigmoid(temp3, forget_output, output->width, sequence);
    


		//gate
		forward(input->data, temp1, input->width, output->width, Uc, sequence, 0);
		forward(cell_output, temp2, input->width, output->width, Wc, sequence, 1);
		element_add(temp1, temp2, temp3, output->width, sequence);
		my_tanh(temp3, gate_output, output->width, sequence);


		//input
		forward(input->data, temp1, input->width, output->width, Ui, sequence, 0);
		forward(cell_output, temp2, input->width, output->width, Wi, sequence, 1);
		element_add(temp1, temp2, temp3, output->width, sequence);
		sigmoid(temp3, input_output, output->width, sequence);
   



		//cell_memory
		element_cell_mul2(cell_memory, forget_output, cell_temp1, output->width, sequence);
		element_mul(gate_output, input_output, cell_temp2, output->width, sequence);
		element_cell_add(cell_temp1, cell_temp2, cell_memory, output->width, sequence);



		//output
		forward(input->data, temp1, input->width, output->width, Uo, sequence, 0);
		forward(cell_output, temp2, input->width, output->width, Wo, sequence, 1);
		element_add(temp1, temp2, temp3, output->width, sequence);
		sigmoid(temp3, output_output, output->width, sequence);



		//cell_output
		my_tanh_cell(cell_memory, cell_temp1, output->width, sequence);
		element_cell_mul(cell_temp1, output_output, cell_output, output->width, sequence);


		// printf("fuck\n"); getchar();

		for (batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (i = 0; i < output->width; i++)
			{
				output->data[batch][output->width*sequence + i] = cell_output[batch][i];
												
			}
		}
	}
}



void forward(float **input, float **output, int input_width, int output_width, float *weight, int sequence, int cell)
{
	int i, j, k, pe, row, batch;
	float sum;
	float temp;


	if (!cell)
	{
		for (batch = 0; batch< BATCH_SIZE; batch++)
		{
			for (i = 0; i < output_width; i++)
			{

				sum = 0.0f;
				for (j = 0; j < input_width; j++)
				{  //vector multiplication
					sum += input[batch][input_width*sequence + j] * weight[(input_width * i) + j];
				}
       
				output[batch][output_width*sequence + i] = sum;
				// printf("%d  %d\n", batch, i);
			}
		}
	}

	else
	{
		for (batch = 0; batch< BATCH_SIZE; batch++)
		{
			for (i = 0; i < output_width; i++)
			{
				sum = 0.0f;
				for (j = 0; j < input_width; j++)
				{                         //vector multiplication
					sum += input[batch][j] * weight[(input_width * i) + j];
				}
				output[batch][output_width*sequence + i] = sum;
			}
		}
	}
}


void load_input(layer *input, const char *file_name[]){
	int batch;
	int file_length;
	FILE *fp;
	file_length = input->sequence * input->width;

	for (batch = 0; batch<input->batch; batch++) {
		fp = fopen(file_name[batch], "r");
		input->data[batch] = (float *)malloc(file_length*sizeof(float));
		printf("batch %d, file_length %d file_name %s \n", batch, file_length, file_name[batch]);
		fread(input->data[batch], sizeof(float), file_length, fp);
		fclose(fp);
	}
}

void load_file(float **w_input, const char *file_name, int f_length) {
	int i;
	FILE *fp;
	if (*w_input == NULL)
		*w_input = (float *)malloc((f_length + 1)*sizeof(float));
	else
		*w_input = (float *)realloc(*w_input, (f_length + 1)*sizeof(float));
	//fp = fopen(file_name, "r");

	//if (fp == NULL)  //file opening error check
	//{
	//	printf("\nerror : can't open file\n\n");
//		exit(1);
//	}

//	fread(*w_input, sizeof(float), f_length, fp);
	//fclose(fp);
	//printf("%s : %f\n", file_name, w_input[0][f_length-1]);
}


void fc_layer(layer *input, layer *output, float *weight, float *bias, int no_relu) {

	int i, j, pe, row, k;
	int batch, sequence;
	float sum;

	for (sequence = 0; sequence < input->sequence; sequence++)
	{
		forward(input->data, output->data, input->width, output->width, weight, sequence, 0);

		for (batch = 0; batch < input->batch; batch++)
		{
			for (i = 0; i < output->width; i++)
			{
				output->data[batch][input->width * sequence + i] = output->data[batch][input->width * sequence + i] + bias[i]; //add bias 
			}
		}
	}
}




void sigmoid(float **input, float **output, int output_width, int sequence)
{
	int i;
	int batch;

	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = exp(input[batch][output_width*sequence + i]) / (exp(input[batch][output_width*sequence + i]) + 1);

}


void my_tanh(float **input, float **output, int output_width, int sequence)
{
	int i;
	int batch;

	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = (exp(2 * input[batch][output_width*sequence + i]) - 1) / (exp(2 * input[batch][output_width*sequence + i]) + 1);

}

void my_tanh_cell(float **input, float **output, int output_width, int sequence)
{
	int i;
	int batch;

	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = (exp(2 * input[batch][i]) - 1) / (exp(2 * input[batch][i]) + 1);

}


void element_add(float **input_a, float **input_b, float **output, int output_width, int sequence)
{
	int i, batch;

	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = input_a[batch][output_width*sequence + i] + input_b[batch][output_width*sequence + i];

}

void element_mul(float **input_a, float **input_b, float **output, int output_width, int sequence)
{
	int i, batch;
	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = input_a[batch][output_width*sequence + i] * input_b[batch][output_width*sequence + i];

}

void element_cell_add(float **input_a, float **input_b, float **output, int output_width, int sequence)
{
	int i, batch;

	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][i] = input_a[batch][output_width*sequence + i] + input_b[batch][output_width*sequence + i];
}


void element_cell_mul(float **input_a, float **input_b, float **output, int output_width, int sequence)
{
	int i, batch;
	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][i] = input_a[batch][output_width*sequence + i] * input_b[batch][output_width*sequence + i];

}

void element_cell_mul2(float **input_a, float **input_b, float **output, int output_width, int sequence)
{
	int i, batch;
	for (batch = 0; batch < BATCH_SIZE; batch++)
	for (i = 0; i < output_width; i++)
		output[batch][output_width*sequence + i] = input_a[batch][i] * input_b[batch][output_width*sequence + i];

}


void evening(layer *input, layer *output)
{
	int i, batch;
   
  for(batch = 0; batch < output->batch; batch++)
  { 
  	for (i = 0; i< input->width; i++)
  	{
  		if(input->data[batch][i] < .0001)
  			output->data[batch][i] = 0.0;
  		else
  			output->data[batch][i] = input->data[0][i];

     }
  }  
}


void softmax(layer *input, layer *output)
{
  int i, batch;
  float sum;
  float largest;
  float e;
  
  
  for(batch = 0; batch < output->batch; batch++)
  {
      sum = 0;
      largest = -340282346638528859811704183484516925440.000000;
      
      for(i = 0; i < output->width; ++i)
          if(input->data[batch][i] > largest) largest = input->data[batch][i];
          
      for(i = 0; i < output->width; ++i)
      {
          e = exp(input->data[batch][i]/0.7 - largest/0.7);
          sum += e;
          output->data[batch][i] = e;
      }
      
      for(i = 0; i < output->width; ++i)
	  {
          output->data[batch][i] /= sum;
	  }
  }
}

void save_symbol(layer *input, layer *output, int sequence, float **symbol_data)
{
	int i, j, batch;

	for(batch = 0; batch < input->batch; batch++)
	{
		symbol_data[batch][sequence] = input->data[batch][0];
		
		for(i = 0; i < input->width; i++)
			output->data[batch][i] = input->data[batch][i];
	}
	
	
}

void print_symbol(float **symbol_data)
{
  int i, batch;
  
  printf("\n\n");
  
  for(batch = 0; batch < BATCH_SIZE; batch++)
  {
    printf("batch: %d\n", batch);
  
    for(i = 0; i < GENERATE_LENGTH; i++)
    {
      printf("%f", symbol_data[batch][i]);
    }
    
    printf("\n\n");
  }
}

void generate_sample_input(layer *input)
{
	int i;
	int batch;
	int file_length = input->sequence * input->width;


	for (batch = 0; batch < input->batch; batch++)
	{
		for (i = 0; i < file_length; i++)
		{
			input->data[batch][i] = pow(-1, rand()) * (rand() % 100);
		}
	}

}