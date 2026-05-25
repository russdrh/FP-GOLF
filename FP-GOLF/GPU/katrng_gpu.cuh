//
//  katrng.h
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//

#ifndef katrng_h_GPU
#define katrng_h_GPU

#include <stdio.h>
#include "../CPU/typed.h"

#define RNG_SUCCESS      0
#define RNG_BAD_MAXLEN  -1
#define RNG_BAD_OUTBUF  -2
#define RNG_BAD_REQ_LEN -3


__device__ void
AES256_CTR_DRBG_Update_gpu(unsigned char *provided_data,
                       unsigned char *Key,
                       unsigned char *V);


__device__ int
seedexpander_init_gpu(AES_XOF_struct *ctx,
                  unsigned char *seed,
                  unsigned char *diversifier,
                  unsigned long maxlen);

__device__ int
seedexpander_gpu(AES_XOF_struct *ctx, unsigned char *x, unsigned long xlen);

__device__ extern void
randombytes_init_gpu(unsigned char *entropy_input,
                 unsigned char *personalization_string,
                 int security_strength);

__device__ extern int
randombytes_gpu(unsigned char *x, unsigned long long xlen);

#endif /* katrng_h */
