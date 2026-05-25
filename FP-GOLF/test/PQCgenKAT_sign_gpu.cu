
//
//  PQCgenKAT_sign.c
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
// #include <helper_cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../GPU/katrng_gpu.cuh"
#include "../GPU/api_gpu.cuh"
#include "../CPU/katrng.h"
#include "../CPU/api.h"


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

template<typename T>
void check(T err, const char* const func, const char* const file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(1);
    }
}

#define	MAX_MARKER_LEN		50

#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4
#define TESTTIMES           3000
#define GRIDSIZE            256
#define BLOCKSIZE           256

char    AlgName[] = "My Alg Name";

#define STR(x)    STR_(x)
#define STR_(x)   #x
// void ReadHex(unsigned char* msg_hex, unsigned char *A, int Length);
void cudainit();
bool comparechar(unsigned char *src1, unsigned int src1Length, unsigned char *src2, unsigned int src2Length);
// bool comparefpr(fpr *src1, unsigned int src1Length, fpr *src2, unsigned int src2Length);

int
main()
{
    cudainit();
    unsigned char       /* *m , */ *sm_c, *sm_g, *sm /*, *m1_c, *m1_g, *m1 */; 
    unsigned long long  mlen = 0, *mlen_c, *mlen_g,  
                        smlen = 0, *smlen_c, *smlen_g /*, mlen1 */;
    double              time, timeUsed, Tput;
    int                 crypto_ret_val;
    bool                test;
    /*
     * Temporary buffers made static to save space on constrained
     * systems (e.g. ARM Cortex M4).
     */
    // static unsigned char       seed[48];
    // static unsigned char       entropy_input[48];
    // static unsigned char       msg[3300];
    // static unsigned char       pk[CRYPTO_PUBLICKEYBYTES], sk[CRYPTO_SECRETKEYBYTES];
    
    static unsigned char       *seed, *seed_c, *seed_g, *nonce_c, *nonce_g;

    unsigned char       *msg_c, *msg_g, *msg;
    unsigned char       *pk_c, *pk_g, *sk_c, *sk_g;
    double              *expanded_key_g;

    seed = (unsigned char*) malloc(48 * sizeof(unsigned char));
    seed_c = (unsigned char*) malloc(48 * sizeof(unsigned char));
    nonce_c = (unsigned char*) malloc(40 * sizeof(unsigned char));
    msg_c = (unsigned char*) malloc(3300 * sizeof(unsigned char));
    msg = (unsigned char*) malloc(3300 * sizeof(unsigned char));
    pk_c = (unsigned char*) malloc(CRYPTO_PUBLICKEYBYTES * sizeof(unsigned char));
    sk_c = (unsigned char*) malloc(CRYPTO_SECRETKEYBYTES * sizeof(unsigned char));
    smlen_c = (unsigned long long*) malloc(64);
    mlen_c = (unsigned long long*) malloc(64);
    if ( (crypto_ret_val = crypto_sign_keypair(pk_c,sk_c)) != 0) {
        printf("crypto_sign_keypair returned <%d>\n", crypto_ret_val);
        return 0;
    }

    static unsigned char       *entropy_input;
    entropy_input = (unsigned char*) malloc(48 * sizeof(unsigned char));
	for (int i=0; i<48; i++){
        entropy_input[i] = i;		
	}
	randombytes_init(entropy_input, NULL, 256);

    checkCudaErrors(cudaMalloc((void**)&seed_g, 48 * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&msg_g, 3300 * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&sk_g, CRYPTO_SECRETKEYBYTES * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&smlen_g, 64 * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&nonce_g, 40 * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&pk_g, CRYPTO_PUBLICKEYBYTES * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&mlen_g, 64 * GRIDSIZE * BLOCKSIZE));
    checkCudaErrors(cudaMalloc((void**)&expanded_key_g, 7169 * sizeof(double) * GRIDSIZE * BLOCKSIZE));
    // checkCudaErrors(cudaMalloc((void**)&b, 128 * 256 * 72 * 512 * sizeof(uint8_t) * GRIDSIZE * BLOCKSIZE));


    


    randombytes(seed, 48);
    mlen = 33*(50+1);
    randombytes(msg_c, mlen);
    msg = msg_c;
    randombytes_init(seed, NULL, 256);
    randombytes(seed_c, 48);
    randombytes(nonce_c, 40);
    // checkCudaErrors(cudaMemcpy(expanded_key_g, expanded_key_c, 7169 * sizeof(double), cudaMemcpyHostToDevice));
    for (int i = 0 ; i < GRIDSIZE * BLOCKSIZE; i++) {
        checkCudaErrors(cudaMemcpy(seed_g + 48 * sizeof(unsigned char) * i, seed_c, 48 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(nonce_g + 40 * sizeof(unsigned char) * i, nonce_c, 40 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(msg_g + 3300 * sizeof(unsigned char) * i, msg_c, 3300 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(sk_g + CRYPTO_SECRETKEYBYTES * sizeof(unsigned char) * i, sk_c, CRYPTO_SECRETKEYBYTES * sizeof(unsigned char), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pk_g + CRYPTO_PUBLICKEYBYTES * sizeof(unsigned char) * i, pk_c, CRYPTO_PUBLICKEYBYTES * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }
    struct timeval real_start, real_stop;

    //SIGNCORRECTNESSTEST - GPU
    sm_c = (unsigned char *)calloc(mlen+CRYPTO_BYTES, sizeof(unsigned char));
    checkCudaErrors(cudaMalloc((void**)&sm_g, (mlen + CRYPTO_BYTES) * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
    // printf("aaa 1 ");
    crypto_expand_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sk_g, expanded_key_g);
    crypto_sign_tree_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sm_g, smlen_g, msg_g, mlen, sk_g, seed_g, nonce_g, expanded_key_g);
    // crypto_sign_dyn_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sm_g, smlen_g, msg_g, mlen, sk_g, seed_g, nonce_g, expanded_key_g);
    // printf(" 2 aaa");
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(sm_c, sm_g, (mlen + CRYPTO_BYTES) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(smlen_c, smlen_g, 64, cudaMemcpyDeviceToHost));
    smlen = *smlen_c;
    //SIGNCORRECTNESSTEST - CPU
    sm = (unsigned char *)calloc(mlen+CRYPTO_BYTES, sizeof(unsigned char));
    if((crypto_ret_val = crypto_sign(sm, &smlen, msg, mlen, sk_c, seed_c, nonce_c)) != 0)
    {
        printf("crypto_sign returned <%d>\n", crypto_ret_val);
        return KAT_CRYPTO_FAILURE;
    }
    test = comparechar(sm_c, *smlen_c, sm, smlen);
    if(test)
    {
        printf("result is right\n");
    }
    else{return 0;}

    time = 0;
    for(int i = 0; i < TESTTIMES; i++)
    {
        // checkCudaErrors(cudaMalloc((void**)&sm_g,(mlen + CRYPTO_BYTES) * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
        gettimeofday(&real_start, NULL);
        // crypto_expand(sk_c, expanded_key_c);
        // printf("%d\n", i);
        // checkCudaErrors(cudaMemcpy(expanded_key_g, expanded_key_c, 7169 * sizeof(double), cudaMemcpyHostToDevice));
        crypto_expand_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sk_g, expanded_key_g);
        // printf("%d\n", i);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&real_stop, NULL);
	    timeUsed = (double) (real_stop.tv_sec - real_start.tv_sec) + (double) (real_stop.tv_usec - real_start.tv_usec) / 1000000;
	    timeUsed = timeUsed * 1000;
        time += timeUsed;
    }
    Tput = (double) GRIDSIZE * BLOCKSIZE * TESTTIMES * 1000 / time;
    printf("Expand: GPU: OUT TimeUsed: %f ms, Throughput: %f ops/sec, Latency: %f ms.\n", time, Tput, time / TESTTIMES);

    //SIGNTIMETEST
    time = 0;
    for(int i = 0; i < TESTTIMES; i++)
    {
        // checkCudaErrors(cudaMalloc((void**)&sm_g,(mlen + CRYPTO_BYTES) * sizeof(unsigned char) * GRIDSIZE * BLOCKSIZE));
        gettimeofday(&real_start, NULL);
        // crypto_expand(sk_c, expanded_key_c);
        // printf("%d\n", i);
        // checkCudaErrors(cudaMemcpy(expanded_key_g, expanded_key_c, 7169 * sizeof(double), cudaMemcpyHostToDevice));
        crypto_sign_tree_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sm_g, smlen_g, msg_g, mlen, sk_g, seed_g, nonce_g, expanded_key_g);
        // crypto_sign_dyn_gpu<<<GRIDSIZE, BLOCKSIZE>>>(sm_g, smlen_g, msg_g, mlen, sk_g, seed_g, nonce_g, expanded_key_g);
        // printf("%d\n", i);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&real_stop, NULL);
	    timeUsed = (double) (real_stop.tv_sec - real_start.tv_sec) + (double) (real_stop.tv_usec - real_start.tv_usec) / 1000000;
	    timeUsed = timeUsed * 1000;
        time += timeUsed;
    }

    Tput = (double) GRIDSIZE * BLOCKSIZE * TESTTIMES * 1000 / time;
	printf("Sign: GPU: OUT TimeUsed: %f ms, Throughput: %f ops/sec, Latency: %f ms.\n", time, Tput, time / TESTTIMES);
    
    time = 0;
    for(int i = 0; i < TESTTIMES; i++)
    {
        sm = (unsigned char *)calloc(mlen+CRYPTO_BYTES, sizeof(unsigned char));
        gettimeofday(&real_start, NULL);
        if((crypto_ret_val = crypto_sign(sm, &smlen, msg, mlen, sk_c, seed_c, nonce_c)) != 0)
        {
            printf("crypto_sign returned <%d>\n", crypto_ret_val);
            return KAT_CRYPTO_FAILURE;
        }
        gettimeofday(&real_stop, NULL);
        timeUsed = (double) (real_stop.tv_sec - real_start.tv_sec) + (double) (real_stop.tv_usec - real_start.tv_usec) / 1000000;
	    timeUsed = timeUsed * 1000;
        time += timeUsed;
    }
    Tput = (double) TESTTIMES * 1000 / time;
	printf("Sign:CPU: OUT TimeUsed: %f ms, Throughput: %f ops/sec, Latency: %f ms.\n", time, Tput, time / TESTTIMES);

    // VRFYCORRECTNESS - GPU
    // checkCudaErrors(cudaMemcpy(sm_g, sm_c, (mlen + CRYPTO_BYTES) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    crypto_sign_open_gpu<<<GRIDSIZE, BLOCKSIZE>>>(msg_g, mlen_g, sm_g, smlen, pk_g);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(mlen_c, mlen_g, 8, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(msg_c, msg_g, 3300, cudaMemcpyDeviceToHost));
    

    // VRFYCORRECTNESS - CPU
    if((crypto_ret_val = crypto_sign_open(msg, &mlen, sm, smlen, pk_c)) != 0)
    {
        printf("crypto_sign returned <%d>\n", crypto_ret_val);
        return KAT_CRYPTO_FAILURE;
    }
    test = comparechar(msg_c, *mlen_c, msg, mlen);
    if(test)
    {
        printf("result is right\n");
    }
    // VRFYTIMETEST
    time = 0;
    for(int i = 0; i < TESTTIMES; i++)
    {
        // checkCudaErrors(cudaMemcpy(sm_g, sm_c, (mlen + CRYPTO_BYTES) * sizeof(unsigned char), cudaMemcpyHostToDevice));
        gettimeofday(&real_start, NULL);
        crypto_sign_open_gpu<<<GRIDSIZE, BLOCKSIZE>>>(msg_g, mlen_g, sm_g, smlen, pk_g);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&real_stop, NULL);
	    timeUsed = (double) (real_stop.tv_sec - real_start.tv_sec) + (double) (real_stop.tv_usec - real_start.tv_usec) / 1000000;
	    timeUsed = timeUsed * 1000;
        time += timeUsed;
    }

    Tput = (double) BLOCKSIZE * GRIDSIZE * TESTTIMES * 1000 / time;
	printf("Veri: GPU: OUT TimeUsed: %f ms, Throughput: %f ops/sec, Latency: %f ms.\n", time, Tput, time / TESTTIMES);
    
    time = 0;
    for(int i = 0; i < TESTTIMES; i++)
    {
        gettimeofday(&real_start, NULL);
        if((crypto_ret_val = crypto_sign_open(msg, &mlen, sm, smlen, pk_c)) != 0)
        {
            printf("crypto_sign returned <%d>\n", crypto_ret_val);
            return KAT_CRYPTO_FAILURE;
        }
        gettimeofday(&real_stop, NULL);
        timeUsed = (double) (real_stop.tv_sec - real_start.tv_sec) + (double) (real_stop.tv_usec - real_start.tv_usec) / 1000000;
	    timeUsed = timeUsed * 1000;
        time += timeUsed;
    }
    Tput = (double) TESTTIMES * 1000 / time;
	printf("Veri: CPU: OUT TimeUsed: %f ms, Throughput: %f ops/sec, Latency: %f ms.\n", time, Tput, time / TESTTIMES);
    return KAT_SUCCESS;
}

void cudainit() {
	int count, i;
	cudaGetDeviceCount(&count);
	//printf("cuda count=%d\n",count);
	//printerror1("get count");
	cudaDeviceProp prop;
	for (i = 0; i < count; i++) {

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				//break;
				//printf("name:%s\n",prop.name);
				;
			}
		}
	}

	for (i = 0; i < count; i++) {

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	printf("select cuda device %d\n", i);
	cudaError_t error_t = cudaSetDevice(i);
	if (error_t != cudaSuccess)
		printf("cuda error\n");

	error_t = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	if (error_t != cudaSuccess)
		printf("cuda error\n");
}

bool comparechar(unsigned char *src1, unsigned int src1Length, unsigned char *src2,
		unsigned int src2Length) {
    bool result = true;
	if (src1Length != src2Length || src1Length == 0) {
		printf("\n src1Length = %d, src2Length = %d\n", src1Length, src2Length);
		return false;
	}
	for (int i = 0; i < src1Length; i++) {
		if (src1[i] != src2[i]) {
			printf("\nsrc %d th, src1 = %d, src2 = %d\n",i,src1[i],src2[i]);
			result = false;
		}
	}
	return result;
}

// bool comparefpr(fpr *src1, unsigned int src1Length, fpr *src2,
// 		unsigned int src2Length) {
//     bool result = true;
// 	if (src1Length != src2Length || src1Length == 0) {
// 		printf("\n src1Length = %d, src2Length = %d\n", src1Length, src2Length);
// 		return false;
// 	}
// 	for (int i = 0; i < src1Length; i++) {
// 		if (src1[i] != src2[i]) {
// 			printf("\nsrc %d th, src1 = %llu, src2 = %llu\n",i,src1[i],src2[i]);
// 			result = false;
// 		}
// 	}
// 	return result;
// }