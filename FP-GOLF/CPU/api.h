#define CRYPTO_SECRETKEYBYTES   1281
#define CRYPTO_PUBLICKEYBYTES   897
#define CRYPTO_BYTES            690
#define CRYPTO_ALGNAME          "Falcon-512"

#define	MAX_MARKER_LEN		50

#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4
#include "typed.h"

int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);

int crypto_sign(unsigned char *sm, unsigned long long *smlen,
	const unsigned char *m, unsigned long long mlen,
	const unsigned char *sk, const unsigned char *seed_c, const unsigned char *nonce_c);

int crypto_sign_open(unsigned char *m, unsigned long long *mlen,
	const unsigned char *sm, unsigned long long smlen,
	const unsigned char *pk);

int crypto(int ret_val, unsigned long long mlen1, unsigned char *m1, 
    unsigned char *pk, unsigned char *sk, unsigned char *sm, 
	unsigned long long smlen, const unsigned char *msg, unsigned long long mlen);

int crypto_expand(const unsigned char *sk, fpr *expanded_key);

