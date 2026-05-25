/*
 * Wrapper for implementing the NIST API for the PQC standardization
 * process.
 */

#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "api_gpu.cuh"
#include "inner_gpu.cuh"

#define NONCELEN   40
#define SEEDLEN    48

/*
 * If stack usage is an issue, define TEMPALLOC to static in order to
 * allocate temporaries in the data section instead of the stack. This
 * would make the crypto_sign_keypair_gpu(), crypto_sign_gpu(), and
 * crypto_sign_open_gpu() functions not reentrant and not thread-safe, so
 * this should be done only for testing purposes.
 */
#define TEMPALLOC

__device__ void randombytes_init_gpu(unsigned char *entropy_input,
	unsigned char *personalization_string,
	int security_strength);
__device__ int randombytes_gpu(unsigned char *x, unsigned long long xlen);


__global__ void
crypto_sign_dyn_gpu(unsigned char *sm, unsigned long long *smlen,
	const unsigned char *m, unsigned long long mlen,
	const unsigned char *sk, const unsigned char *seed_g, const unsigned char *nonce_g, double *expanded_key)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TEMPALLOC union {
		uint8_t b[72 * 512];
		uint64_t dummy_u64;
		fpr dummy_fpr;
	} tmp;
	TEMPALLOC int8_t f[512], g[512], F[512], G[512];
	TEMPALLOC union {
		int16_t sig[512];
		uint16_t hm[512];
	} r;
	TEMPALLOC unsigned char esig[CRYPTO_BYTES - 2 - NONCELEN];
	TEMPALLOC inner_shake256_context sc;
	size_t u, v, sig_len;

    __shared__ fpr gm_tab_sh[2048];
	int td = threadIdx.x;
    for (int i = td; i < 2048; i += blockDim.x) {
        gm_tab_sh[i] = fpr_gm_tab_gpu[i];
    }
    __syncthreads();

	/*
	 * Decode the private key.
	 */
	if (sk[0 + tid * CRYPTO_SECRETKEYBYTES] != 0x50 + 9) {
		return;
	}
	u = 1;
	v = Zf(trim_i8_decode_gpu)(f, 9, Zf(max_fg_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	v = Zf(trim_i8_decode_gpu)(g, 9, Zf(max_fg_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	v = Zf(trim_i8_decode_gpu)(F, 9, Zf(max_FG_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	if (u != CRYPTO_SECRETKEYBYTES) {
		return;
	}
	if (!Zf(complete_private_gpu)(G, f, g, F, 9, tmp.b)) {
		return;
	}
	
	/*
	 * Hash message nonce + message into a vector.
	 */
	inner_shake256_init(&sc);
	inner_shake256_inject(&sc, nonce_g, NONCELEN);
	inner_shake256_inject(&sc, m + tid * 3300, mlen);
	inner_shake256_flip(&sc);
	Zf(hash_to_point_vartime_gpu)(&sc, r.hm, 9);
	/*
	 * Initialize a RNG.
	 */
	inner_shake256_init(&sc);
	inner_shake256_inject(&sc, seed_g + tid * 48, SEEDLEN);
	inner_shake256_flip(&sc);
	/*
	 * Compute the signature.
	 */
	
	
	Zf(sign_dyn_gpu)(r.sig, &sc, f, g, F, G, r.hm, 9, tmp.b, gm_tab_sh);

	
	/*
	 * Encode the signature and bundle it with the message. Format is:
	 *   signature length     2 bytes, big-endian
	 *   nonce                40 bytes
	 *   message              mlen bytes
	 *   signature            slen bytes
	 */
	esig[0] = 0x20 + 9;
	sig_len = Zf(comp_encode_gpu)(esig + 1, (sizeof esig) - 1, r.sig, 9);
	if (sig_len == 0) {
		return;
	}
	sig_len ++;
	my_memmove_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2 + NONCELEN, m, mlen);
	sm[0 + tid * (mlen + CRYPTO_BYTES)] = (unsigned char)(sig_len >> 8);
	sm[1 + tid * (mlen + CRYPTO_BYTES)] = (unsigned char)sig_len;
	my_memcpy_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2, nonce_g, NONCELEN);
	my_memcpy_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2 + NONCELEN + mlen, esig, sig_len);
	*(smlen + tid) = 2 + NONCELEN + mlen + sig_len;
	return;
}

__global__ void
crypto_sign_tree_gpu(unsigned char *sm, unsigned long long *smlen,
	const unsigned char *m, unsigned long long mlen,
	const unsigned char *sk, const unsigned char *seed_g, const unsigned char *nonce_g, double *expanded_key)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TEMPALLOC union {
		uint8_t b[72 * 512];
		uint64_t dummy_u64;
		fpr dummy_fpr;
	} tmp;
	TEMPALLOC union {
		int16_t sig[512];
		uint16_t hm[512];
	} r;
	TEMPALLOC unsigned char esig[CRYPTO_BYTES - 2 - NONCELEN];
	TEMPALLOC inner_shake256_context sc;
	size_t sig_len;

	__shared__ fpr gm_tab_sh[2048];
	int td = threadIdx.x;
    for (int i = td; i < 2048; i += blockDim.x) {
        gm_tab_sh[i] = fpr_gm_tab_gpu[i];
    }
    __syncthreads();
	/*
	 * Hash message nonce + message into a vector.
	 */
	inner_shake256_init(&sc);
	inner_shake256_inject(&sc, nonce_g, NONCELEN);
	inner_shake256_inject(&sc, m + tid * 3300, mlen);
	inner_shake256_flip(&sc);
	Zf(hash_to_point_vartime_gpu)(&sc, r.hm, 9);
	/*
	 * Initialize a RNG.
	 */
	inner_shake256_init(&sc);
	inner_shake256_inject(&sc, seed_g + tid * 48, SEEDLEN);
	inner_shake256_flip(&sc);
	/*
	 * Compute the signature.
	 */
	Zf(sign_tree_gpu)(r.sig, &sc, expanded_key + tid * 7169, r.hm, 9, tmp.b, gm_tab_sh);
	/*
	 * Encode the signature and bundle it with the message. Format is:
	 *   signature length     2 bytes, big-endian
	 *   nonce                40 bytes
	 *   message              mlen bytes
	 *   signature            slen bytes
	 */
	esig[0] = 0x20 + 9;
	sig_len = Zf(comp_encode_gpu)(esig + 1, (sizeof esig) - 1, r.sig, 9);
	if (sig_len == 0) {
		return;
	}
	sig_len ++;
	my_memmove_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2 + NONCELEN, m, mlen);
	sm[0 + tid * (mlen + CRYPTO_BYTES)] = (unsigned char)(sig_len >> 8);
	sm[1 + tid * (mlen + CRYPTO_BYTES)] = (unsigned char)sig_len;
	my_memcpy_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2, nonce_g, NONCELEN);
	my_memcpy_gpu(sm + tid * (mlen + CRYPTO_BYTES) + 2 + NONCELEN + mlen, esig, sig_len);
	*(smlen + tid) = 2 + NONCELEN + mlen + sig_len;
	return;
}

__global__ void
crypto_expand_gpu(const unsigned char *sk, fpr *expanded_key)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TEMPALLOC uint8_t b[72 * 512];
	TEMPALLOC int8_t f[512], g[512], F[512], G[512];
	size_t u, v;

	__shared__ fpr gm_tab_sh[2048];
	int td = threadIdx.x;
    for (int i = td; i < 2048; i += blockDim.x) {
        gm_tab_sh[i] = fpr_gm_tab_gpu[i];
    }
    __syncthreads();
	/*
	 * Decode the private key.
	 */
	if (sk[0 + tid * CRYPTO_SECRETKEYBYTES] != 0x50 + 9) {
		return;
	}
	u = 1;
	v = Zf(trim_i8_decode_gpu)(f, 9, Zf(max_fg_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	v = Zf(trim_i8_decode_gpu)(g, 9, Zf(max_fg_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	v = Zf(trim_i8_decode_gpu)(F, 9, Zf(max_FG_bits_gpu)[9],
		sk + tid * CRYPTO_SECRETKEYBYTES + u, CRYPTO_SECRETKEYBYTES - u);
	if (v == 0) {
		return;
	}
	u += v;
	if (u != CRYPTO_SECRETKEYBYTES) {
		return;
	}


	if (!Zf(complete_private_gpu)(G, f, g, F, 9, b)) {
		return;
	}
	Zf(expand_privkey_gpu)(expanded_key + 7169 * tid, f, g, F, G, 9, b, gm_tab_sh);
	return;
}

__global__ void
crypto_sign_open_gpu(unsigned char *m, unsigned long long *mlen,
	const unsigned char *sm, unsigned long long smlen,
	const unsigned char *pk)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TEMPALLOC union {
		uint8_t b[2 * 512];
		uint64_t dummy_u64;
		fpr dummy_fpr;
	} tmp;
	const unsigned char *esig;
	TEMPALLOC uint16_t h[512], hm[512];
	TEMPALLOC int16_t sig[512];
	TEMPALLOC inner_shake256_context sc;
	size_t sig_len, msg_len;

	/*
	 * Decode public key.
	 */
	if (pk[0 + tid * CRYPTO_PUBLICKEYBYTES] != 0x00 + 9) {
		return;
	}
	if (Zf(modq_decode_gpu)(h, 9, pk + tid * CRYPTO_PUBLICKEYBYTES + 1, CRYPTO_PUBLICKEYBYTES - 1)
		!= CRYPTO_PUBLICKEYBYTES - 1)
	{
		return;
	}
	Zf(to_ntt_monty_gpu)(h, 9);

	/*
	 * Find nonce, signature, message length.
	 */
	if (smlen < 2 + NONCELEN) {
		return;
	}
	sig_len = ((size_t)sm[0 + tid * (*mlen + CRYPTO_BYTES)] << 8) | (size_t)sm[1 + tid * (*mlen + CRYPTO_BYTES)];
	if (sig_len > (smlen - 2 - NONCELEN)) {
		return;
	}
	msg_len = smlen - 2 - NONCELEN - sig_len;

	/*
	 * Decode signature.
	 */
	esig = sm + tid * (*mlen + CRYPTO_BYTES) + 2 + NONCELEN + msg_len;
	if (sig_len < 1 || esig[0] != 0x20 + 9) {
		return;
	}
	if (Zf(comp_decode_gpu)(sig, 9,
		esig + 1, sig_len - 1) != sig_len - 1)
	{
		return;
	}

	/*
	 * Hash nonce + message into a vector.
	 */
	inner_shake256_init(&sc);
	inner_shake256_inject(&sc, sm + tid * (*mlen + CRYPTO_BYTES) + 2, NONCELEN + msg_len);
	inner_shake256_flip(&sc);
	Zf(hash_to_point_vartime_gpu)(&sc, hm, 9);

	/*
	 * Verify signature.
	 */
	if (!Zf(verify_raw_gpu)(hm, sig, h, 9, tmp.b)) {
		return;
	}

	/*
	 * Return plaintext.
	 */
	my_memmove_gpu(m, sm + tid * (*mlen + CRYPTO_BYTES) + 2 + NONCELEN, msg_len);
	*(mlen + tid) = msg_len;
	return;
}
