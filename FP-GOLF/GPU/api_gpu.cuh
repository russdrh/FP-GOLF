#define CRYPTO_SECRETKEYBYTES   1281
#define CRYPTO_PUBLICKEYBYTES   897
#define CRYPTO_BYTES            690
#define CRYPTO_ALGNAME          "Falcon-512"

#define	MAX_MARKER_LEN		50

#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4
#include "../CPU/typed.h"

__device__ int crypto_sign_keypair_gpu(unsigned char *pk, unsigned char *sk);

__global__ void crypto_expand_gpu(const unsigned char *sk, fpr *expanded_key);

__global__ void crypto_sign_tree_gpu(unsigned char *sm, unsigned long long *smlen,
	const unsigned char *m, unsigned long long mlen,
	const unsigned char *sk, const unsigned char *seed_g, const unsigned char *nonce_g, double *expanded_key);

__global__ void crypto_sign_dyn_gpu(unsigned char *sm, unsigned long long *smlen,
	const unsigned char *m, unsigned long long mlen,
	const unsigned char *sk, const unsigned char *seed_g, const unsigned char *nonce_g, double *expanded_key);

__global__ void crypto_sign_open_gpu(unsigned char *m, unsigned long long *mlen,
	const unsigned char *sm, unsigned long long smlen,
	const unsigned char *pk);


