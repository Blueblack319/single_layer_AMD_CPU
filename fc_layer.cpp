#include <stdio.h>

// void fc_layer(size_t data_cnt,	 // D: 32 ~ 4096
// 			  size_t input_dim,	 // N
// 			  size_t output_dim, // M
// 			  float *matrix,
// 			  float *bias,
// 			  float *input,
// 			  float *output)
// {
// 	// loop over input instances
// 	for (size_t iidx = 0; iidx < data_cnt; iidx++)
// 	{
// 		// loop over weight columns
// 		for (size_t oidx = 0; oidx < output_dim; oidx++)
// 		{
// 			float outv = 0;
// 			// loop over each input's activation values
// 			for (size_t aidx = 0; aidx < input_dim; aidx++)
// 			{
// 				float inv = input[input_dim * iidx + aidx];		 // N * d + n
// 				float weight = matrix[output_dim * aidx + oidx]; // M
// 				outv += inv * weight;
// 			}
// 			// Add bias
// 			outv += bias[oidx];
// 			// ReLU activation function
// 			// cmov 가능?
// 			if (outv < 0)
// 				outv = 0;
// 			// Store the output
// 			output[iidx * output_dim + oidx] = outv;
// 		}
// 	}
// }

void fc_layer(size_t data_cnt,	 // D: 32 ~ 4096
			  size_t input_dim,	 // N
			  size_t output_dim, // M
			  float *matrix,
			  float *bias,
			  float *input,
			  float *output)
{
	size_t i, o, a, iidx, oidx, aidx, B = 64;
	size_t cond = input_dim - B;
	for (i = 0; i < data_cnt; i += B)
		for (o = 0; o < output_dim; o += B)
			for (a = 0; a < input_dim; a += B)
				// B x B mini matrix multiplication
				for (oidx = o; oidx < o + B; oidx++)
				{
					for (iidx = i; iidx < i + B; iidx++)
					{
						float outv = output[iidx * output_dim + oidx];
						for (aidx = a; aidx < a + B; aidx++)
						{
							float inv = input[input_dim * iidx + aidx];		 // N * d + n
							float weight = matrix[output_dim * aidx + oidx]; // M
							outv += inv * weight;
						}
						if (a == cond)
						{
							// Add bias
							outv += bias[oidx];
							// ReLU activation function
							// cmov 가능?
							if (outv < 0)
								outv = 0;
						}
						// Store the output
						output[iidx * output_dim + oidx] = outv;
					}
				}
}
