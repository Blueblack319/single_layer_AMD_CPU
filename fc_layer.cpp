#include <stdio.h>
#include <immintrin.h>

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

// void fc_layer1(size_t data_cnt,	  // D: 32 ~ 4096
// 			   size_t input_dim,  // N
// 			   size_t output_dim, // M
// 			   float *matrix,
// 			   float *bias,
// 			   float *input,
// 			   float *output)
// {
// 	size_t i, o, a, iidx, oidx, aidx, B = 64;
// 	size_t cond = input_dim - B;
// 	for (i = 0; i < data_cnt; i += B)
// 		for (o = 0; o < output_dim; o += B)
// 			for (a = 0; a < input_dim; a += B)
// 				// B x B mini matrix multiplication
// 				for (oidx = o; oidx < o + B; oidx++)
// 				{
// 					for (iidx = i; iidx < i + B; iidx++)
// 					{
// 						float outv = output[iidx * output_dim + oidx];
// 						for (aidx = a; aidx < a + B; aidx++)
// 						{
// 							float inv = input[input_dim * iidx + aidx];		 // N * d + n
// 							float weight = matrix[output_dim * aidx + oidx]; // M
// 							outv += inv * weight;
// 						}
// 						if (a == cond)
// 						{
// 							// Add bias
// 							outv += bias[oidx];
// 							// ReLU activation function
// 							// cmov 가능?
// 							if (outv < 0)
// 								outv = 0;
// 						}
// 						// Store the output
// 						output[iidx * output_dim + oidx] = outv;
// 					}
// 				}
// }

// void fc_layer(size_t data_cnt,	 // D: 32 ~ 4096
// 			  size_t input_dim,	 // N
// 			  size_t output_dim, // M
// 			  double *matrix,
// 			  double *bias,
// 			  double *input,
// 			  double *output)
// {
// 	int i, o, a, iidx, oidx, aidx, B = 32;
// 	__m256d outv;
// 	int cond = input_dim - B;

// 	// Transpose

// 	for (i = 0; i < data_cnt; i += B)
// 		for (o = 0; o < output_dim; o += B)
// 		{
// 			for (a = 0; a < input_dim; a += B)
// 				// B x B mini matrix multiplication
// 				for (iidx = i; iidx < i + B; iidx++)
// 				{
// 					for (oidx = o; oidx < o + B; oidx += 8)
// 					{
// 						// outv = output[iidx * output_dim + oidx];
// 						outv = _mm256_load_pd(&output[iidx * output_dim + oidx]);
// 						for (aidx = a; aidx < a + B; aidx += 1)
// 						{
// 							// outv += input[input_dim * iidx + aidx] * matrix[output_dim * aidx + oidx];
// 							__m256d a = _mm256_broadcast_sd(&input[input_dim * iidx + aidx]);
// 							__m256d b = _mm256_load_pd(&matrix[output_dim * aidx + oidx]);
// 							outv = _mm256_fmadd_pd(a, b, outv);
// 							// outv = _mm256_add_ps(outv, _mm256_mul_ps(_mm256_broadcast_ss(&input[input_dim * iidx + aidx]), _mm256_load_ps(&matrix[output_dim * aidx + oidx])));
// 						}
// 						if (a == cond)
// 						{
// 							// outv += bias[oidx];
// 							// outv = _mm256_max_ps(_mm256_add_ps(outv, _mm256_broadcast_ss(bias + oidx)), _mm256_setzero_ps());
// 							outv = _mm256_add_pd(_mm256_broadcast_sd(&bias[oidx]), outv);
// 							outv = _mm256_max_pd(_mm256_setzero_pd(), outv);

// 							// ReLU activation function
// 							// cmov 가능?
// 							// if (outv < 0)
// 							// 	outv = 0;
// 						}
// 						// Store the output
// 						// output[iidx * output_dim + oidx] = outv;
// 						_mm256_store_pd(&output[iidx * output_dim + oidx], outv);
// 					}
// 				}
// 		}
// }

void fc_layer(size_t data_cnt,	 // D: 32 ~ 4096
			  size_t input_dim,	 // N
			  size_t output_dim, // M
			  float *matrix,
			  float *bias,
			  float *input,
			  float *output)
{
	int i, o, a, iidx, oidx, aidx, B = 32;
	__m256 outv;
	int cond = input_dim - B;

	// Transpose

	for (i = 0; i < data_cnt; i += B)
		for (o = 0; o < output_dim; o += B)
		{
			for (a = 0; a < input_dim; a += B)
				// B x B mini matrix multiplication
				for (iidx = i; iidx < i + B; iidx++)
				{
					for (oidx = o; oidx < o + B; oidx += 8)
					{
						// outv = output[iidx * output_dim + oidx];
						outv = _mm256_load_ps(&output[iidx * output_dim + oidx]);
						for (aidx = a; aidx < a + B; aidx += 1)
						{
							// outv += input[input_dim * iidx + aidx] * matrix[output_dim * aidx + oidx];
							__m256 a = _mm256_broadcast_ss(&input[input_dim * iidx + aidx]);
							__m256 b = _mm256_load_ps(&matrix[output_dim * aidx + oidx]);
							outv = _mm256_fmadd_ps(a, b, outv);
							// outv = _mm256_add_ps(outv, _mm256_mul_ps(_mm256_broadcast_ss(&input[input_dim * iidx + aidx]), _mm256_load_ps(&matrix[output_dim * aidx + oidx])));
						}
						if (a == cond)
						{
							// outv += bias[oidx];
							// outv = _mm256_max_ps(_mm256_add_ps(outv, _mm256_broadcast_ss(bias + oidx)), _mm256_setzero_ps());
							outv = _mm256_add_ps(_mm256_broadcast_ss(&bias[oidx]), outv);
							outv = _mm256_max_ps(_mm256_setzero_ps(), outv);

							// ReLU activation function
							// cmov 가능?
							// if (outv < 0)
							// 	outv = 0;
						}
						// Store the output
						// output[iidx * output_dim + oidx] = outv;
						_mm256_store_ps(&output[iidx * output_dim + oidx], outv);
					}
				}
		}
}