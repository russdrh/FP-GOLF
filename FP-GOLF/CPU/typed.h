#include <stdint.h>
#ifndef TYPEDEF_DEFINED

#define TYPEDEF_DEFINED

typedef double fpr;

typedef struct {
	union {
		uint64_t A[25];
		uint8_t dbuf[200];
	} st;
	uint64_t dptr;
} inner_shake256_context;

typedef struct {
	union {
		uint8_t d[512]; /* MUST be 512, exactly */
		uint64_t dummy_u64;
	} buf;
	size_t ptr;
	union {
		uint8_t d[256];
		uint64_t dummy_u64;
	} state;
	int type;
} prng;

typedef struct {
	prng p;
	fpr sigma_min;
} sampler_context;

typedef struct {
    unsigned char   buffer[16];
    int             buffer_pos;
    unsigned long   length_remaining;
    unsigned char   key[32];
    unsigned char   ctr[16];
} AES_XOF_struct;

typedef struct {
    unsigned char   Key[32];
    unsigned char   V[16];
    int             reseed_counter;
} AES256_CTR_DRBG_struct;

typedef struct
 {
	fpr *t0; 
	fpr *g00; 
	fpr *g11;
	unsigned logn; 
	unsigned char is_z0, is_z1;
 } STACK;
#endif
