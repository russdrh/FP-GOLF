/*
 * Falcon signature generation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2017-2019  Falcon Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@nccgroup.com>
 */
#include <stdio.h>
#include "inner_gpu.cuh"
#include <cuda_runtime.h>
#include "../CPU/typed.h"
/* =================================================================== */

/*
 * Compute degree N from logarithm 'logn'.
 */
#define MKN(logn)   ((size_t)1 << (logn))

/* =================================================================== */
/*
 * Binary case:
 *   N = 2^logn
 *   phi = X^N+1
 */

/*
 * Get the size of the LDL tree for an input with polynomials of size
 * 2^logn. The size is expressed in the number of elements.
 */
__device__ static inline unsigned
ffLDL_treesize_gpu(unsigned logn)
{
	/*
	 * For logn = 0 (polynomials are constant), the "tree" is a
	 * single element. Otherwise, the tree node has size 2^logn, and
	 * has two child trees for size logn-1 each. Thus, treesize s()
	 * must fulfill these two relations:
	 *
	 *   s(0) = 1
	 *   s(logn) = (2^logn) + 2*s(logn-1)
	 */
	return (logn + 1) << logn;
}

/*
 * Inner function for ffLDL_fft_gpu(). It expects the matrix to be both
 * auto-adjoint and quasicyclic; also, it uses the source operands
 * as modifiable temporaries.
 *
 * tmp[] must have room for at least one polynomial.
 */

__device__ static void
ffLDL_fft_inner_gpu(fpr *tree, fpr *g0, fpr *g1, unsigned logn, fpr *tmp, const fpr *gm_tab_sh)
{
	typedef struct {
    fpr *g0;
    fpr *g1;
    size_t tree_pos;
    unsigned logn;
	} Node;

    size_t n = MKN(logn);
    if (n == 1) {
        tree[0] = g0[0];
        return;
    }

    Node stack[11]; 
    int sp = 0;

    stack[sp].g0 = g0;
    stack[sp].g1 = g1;
    stack[sp].tree_pos = 0;
    stack[sp++].logn = logn;

    fpr *level_tmp = tmp;  

    while (sp > 0) {
        Node curr = stack[--sp];
        size_t curr_n = (size_t)1 << curr.logn;
        if (curr_n == 1) {
            tree[curr.tree_pos] = curr.g0[0];
            continue;
        }

        size_t hn = curr_n >> 1;

        fpr *d11 = level_tmp;
        Zf(poly_LDLmv_fft_gpu)(d11, tree + curr.tree_pos, curr.g0, curr.g1, curr.g0, curr.logn);

        fpr *left_g0 = level_tmp + curr_n;  
        fpr *left_g1 = left_g0 + hn;
        fpr *right_g0 = left_g1 + hn;
        fpr *right_g1 = right_g0 + hn;

        Zf(poly_split_fft_gpu)(left_g0, left_g1, curr.g0, curr.logn, gm_tab_sh);
        Zf(poly_split_fft_gpu)(right_g0, right_g1, d11, curr.logn, gm_tab_sh);

        unsigned child_logn = curr.logn - 1;
        size_t child_ts = ffLDL_treesize_gpu(child_logn);

        stack[sp].g0 = right_g0;
        stack[sp].g1 = right_g1;
        stack[sp].tree_pos = curr.tree_pos + curr_n + child_ts;
        stack[sp++].logn = child_logn;

        stack[sp].g0 = left_g0;
        stack[sp].g1 = left_g1;
        stack[sp].tree_pos = curr.tree_pos + curr_n;
        stack[sp++].logn = child_logn;
    }
}



/*
 * Compute the ffLDL tree of an auto-adjoint matrix G. The matrix
 * is provided as three polynomials (FFT representation).
 *
 * The "tree" array is filled with the computed tree, of size
 * (logn+1)*(2^logn) elements (see ffLDL_treesize_gpu()).
 *
 * Input arrays MUST NOT overlap, except possibly the three unmodified
 * arrays g00, g01 and g11. tmp[] should have room for at least three
 * polynomials of 2^logn elements each.
 */
__device__ static void
ffLDL_fft_gpu(fpr *tree, const fpr *g00,
	const fpr *g01, const fpr *g11,
	unsigned logn, fpr *tmp, const fpr *gm_tab_sh)
{
	size_t n, hn;
	fpr *d00, *d11;

	n = MKN(logn);
	if (n == 1) {
		tree[0] = g00[0];
		return;
	}
	hn = n >> 1;
	d00 = tmp;
	d11 = tmp + n;
	tmp += n << 1;

	my_memcpy_gpu(d00, g00, n * sizeof *g00);
	Zf(poly_LDLmv_fft_gpu)(d11, tree, g00, g01, g11, logn);

	Zf(poly_split_fft_gpu)(tmp, tmp + hn, g00, logn, gm_tab_sh);
	Zf(poly_split_fft_gpu)(d00, d00 + hn, d11, logn, gm_tab_sh);
	my_memcpy_gpu(d11, tmp, n * sizeof *tmp);
	ffLDL_fft_inner_gpu(tree + n,
		d11, d11 + hn, logn - 1, tmp, gm_tab_sh);
	ffLDL_fft_inner_gpu(tree + n + ffLDL_treesize_gpu(logn - 1),
		d00, d00 + hn, logn - 1, tmp, gm_tab_sh);
}

/*
 * Normalize an ffLDL tree: each leaf of value x is replaced with
 * sigma / sqrt(x).
 */

__device__ static void ffLDL_binary_normalize_gpu(fpr *tree, unsigned orig_logn, unsigned logn) {
    struct Frame { fpr *ptr; unsigned logn; };
    const int MAXSTACK = 64;
    Frame stack[MAXSTACK];
    int top = 0;
    stack[top++] = { tree, logn };

    while (top > 0) {
        Frame cur = stack[--top];
        fpr *ptr = cur.ptr;
        unsigned ln = cur.logn;
        size_t n = MKN(ln);

        if (n == 1) {
            ptr[0] = fpr_mul_gpu(fpr_sqrt_gpu(ptr[0]), fpr_inv_sigma_gpu[orig_logn]);
        } else {
            size_t subtree_size = ffLDL_treesize_gpu(ln - 1);
            stack[top++] = { ptr + n + subtree_size, ln - 1 };
            stack[top++] = { ptr + n, ln - 1 };
        }
    }
}

/* =================================================================== */

/*
 * Convert an integer polynomial (with small values) into the
 * representation with complex numbers.
 */
__device__ static void
smallints_to_fpr_gpu(fpr *r, const int8_t *t, unsigned logn)
{
	size_t n, u;

	n = MKN(logn);
	for (u = 0; u < n; u ++) {
		r[u] = fpr_of_gpu(t[u]);
	}
}

/*
 * The expanded private key contains:
 *  - The B0 matrix (four elements)
 *  - The ffLDL tree
 */

__device__ static inline size_t
skoff_b00_gpu(unsigned logn)
{
	(void)logn;
	return 0;
}

__device__ static inline size_t
skoff_b01_gpu(unsigned logn)
{
	return MKN(logn);
}

__device__ static inline size_t
skoff_b10_gpu(unsigned logn)
{
	return 2 * MKN(logn);
}

__device__ static inline size_t
skoff_b11_gpu(unsigned logn)
{
	return 3 * MKN(logn);
}

__device__ static inline size_t
skoff_tree_gpu(unsigned logn)
{
	return 4 * MKN(logn);
}

/* see inner.h */
__device__ void
Zf(expand_privkey_gpu)(fpr *expanded_key,
	const int8_t *f, const int8_t *g,
	const int8_t *F, const int8_t *G,
	unsigned logn, uint8_t *tmp, const fpr *gm_tab_sh)
{
	unsigned long long t0 = clock64();
	size_t n;
	fpr *rf, *rg, *rF, *rG;
	fpr *b00, *b01, *b10, *b11;
	fpr *g00, *g01, *g11, *gxx;
	fpr *tree;
	
	// unsigned long long defi = clock64();
	// printf("def: %llu\n", defi - t0);

	n = MKN(logn);
	b00 = expanded_key + skoff_b00_gpu(logn);
	b01 = expanded_key + skoff_b01_gpu(logn);
	b10 = expanded_key + skoff_b10_gpu(logn);
	b11 = expanded_key + skoff_b11_gpu(logn);
	tree = expanded_key + skoff_tree_gpu(logn);
	
	// unsigned long long exp = clock64();
	// printf("exp: %llu\n", exp - defi);

	/*
	 * We load the private key elements directly into the B0 matrix,
	 * since B0 = [[g, -f], [G, -F]].
	 */
	rf = b01;
	rg = b00;
	rF = b11;
	rG = b10;

	smallints_to_fpr_gpu(rf, f, logn);
	smallints_to_fpr_gpu(rg, g, logn);
	smallints_to_fpr_gpu(rF, F, logn);
	smallints_to_fpr_gpu(rG, G, logn);
	// unsigned long long smallint = clock64();
	// printf("smallint: %llu\n", smallint - exp);

	/*
	 * Compute the FFT for the key elements, and negate f and F.
	 */
	Zf(FFT_gpu)(rf, logn, gm_tab_sh);
	Zf(FFT_gpu)(rg, logn, gm_tab_sh);
	Zf(FFT_gpu)(rF, logn, gm_tab_sh);
	Zf(FFT_gpu)(rG, logn, gm_tab_sh);
	Zf(poly_neg_gpu)(rf, logn);
	Zf(poly_neg_gpu)(rF, logn);
	
	// unsigned long long fft = clock64();
	// printf("fft: %llu\n", fft - smallint);

	/*
	 * The Gram matrix is G = B·B*. Formulas are:
	 *   g00 = b00*adj(b00) + b01*adj(b01)
	 *   g01 = b00*adj(b10) + b01*adj(b11)
	 *   g10 = b10*adj(b00) + b11*adj(b01)
	 *   g11 = b10*adj(b10) + b11*adj(b11)
	 *
	 * For historical reasons, this implementation uses
	 * g00, g01 and g11 (upper triangle).
	 */
	g00 = (fpr *)tmp;
	g01 = g00 + n;
	g11 = g01 + n;
	gxx = g11 + n;

	my_memcpy_gpu(g00, b00, n * sizeof *b00);
	Zf(poly_mulselfadj_fft_gpu)(g00, logn);
	my_memcpy_gpu(gxx, b01, n * sizeof *b01);
	Zf(poly_mulselfadj_fft_gpu)(gxx, logn);
	Zf(poly_add_gpu)(g00, gxx, logn);

	my_memcpy_gpu(g01, b00, n * sizeof *b00);
	Zf(poly_muladj_fft_gpu)(g01, b10, logn);
	my_memcpy_gpu(gxx, b01, n * sizeof *b01);
	Zf(poly_muladj_fft_gpu)(gxx, b11, logn);
	Zf(poly_add_gpu)(g01, gxx, logn);

	my_memcpy_gpu(g11, b10, n * sizeof *b10);
	Zf(poly_mulselfadj_fft_gpu)(g11, logn);
	my_memcpy_gpu(gxx, b11, n * sizeof *b11);
	Zf(poly_mulselfadj_fft_gpu)(gxx, logn);
	Zf(poly_add_gpu)(g11, gxx, logn);

	/*
	 * Compute the Falcon tree.
	 */
	ffLDL_fft_gpu(tree, g00, g01, g11, logn, gxx, gm_tab_sh);
	/*
	 * Normalize tree.
	 */
	ffLDL_binary_normalize_gpu(tree, logn, logn);
}

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

/*
 * Perform Fast Fourier Sampling for target vector t. The Gram matrix
 * is provided (G = [[g00, g01], [adj(g01), g11]]). The sampled vector
 * is written over (t0,t1). The Gram matrix is modified as well. The
 * tmp[] buffer must have room for four polynomials.
 */
 
__device__ static void
ffSampling_fft_dyntree_gpu(samplerZ samp, void *samp_ctx,
	fpr *t0, fpr *t1,
	fpr *g00, fpr *g01, fpr *g11,
	unsigned orig_logn, unsigned logn, fpr *tmp, const fpr *gm_tab_sh)
{
    size_t n, hn;
	STACK stack[10];
	unsigned stack_top = 0;

	stack[0].t0 = t0;
	stack[0].g00 = g00;
	stack[0].g11 = g11;
	stack[0].logn = logn;
	stack[0].is_z0 = 0;
	stack[0].is_z1 = 0;
	while (1)
	{	
		/*
		 * Deepest level: the LDL tree leaf value is just g00 (the
		 * array has length only 1 at this point); we normalize it
		 * with regards to sigma, then use it for sampling.
		 */
		if (stack[stack_top].logn == 0) {
			fpr leaf;

			leaf = stack[stack_top].g00[0];
			leaf = fpr_mul_gpu(fpr_sqrt_gpu(stack[stack_top].g00[0]), fpr_inv_sigma_gpu[orig_logn]);
			stack[stack_top].t0[0] = (double)(samp(samp_ctx, stack[stack_top].t0[0], leaf));
			stack[stack_top].t0[1] = (double)(samp(samp_ctx, stack[stack_top].t0[1], leaf));
			if (!stack[--stack_top].is_z0)
			{
				Zf(poly_merge_fft_gpu)(stack[stack_top].t0 + 8, stack[stack_top].t0 + 6, stack[stack_top].t0 + 7, 1, gm_tab_sh);
			}
			else
			{
				Zf(poly_merge_fft_gpu)(stack[stack_top].t0, stack[stack_top].t0 + 4, stack[stack_top].t0 + 5, 1, gm_tab_sh);
			}
		}
		else
		{
			n = (size_t)1 << stack[stack_top].logn;
			hn = n >> 1;

			if (!stack[stack_top].is_z1)
			{
				/*
				 * Decompose G into LDL. We only need d00 (identical to g00),
				 * d11, and l10; we do that in place.
				 */
				Zf(poly_LDL_fft_gpu)(stack[stack_top].g00, stack[stack_top].g00 + n, stack[stack_top].g11, stack[stack_top].logn);
				/*
				 * Split d00 and d11 and expand them into half-size quasi-cyclic
				 * Gram matrices. We also save l10 in tmp[].
				 */
				Zf(poly_split_fft_gpu)(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].g00, stack[stack_top].logn, gm_tab_sh);	
				my_memcpy_gpu(stack[stack_top].g00, stack[stack_top].t0 + (n << 1), n * sizeof *(t0 + (n << 1)));
				Zf(poly_split_fft_gpu)(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].g11, stack[stack_top].logn, gm_tab_sh);
				my_memcpy_gpu(stack[stack_top].g11, stack[stack_top].t0 + (n << 1), n * sizeof *(t0 + (n << 1)));
				my_memcpy_gpu(stack[stack_top].t0 + (n << 1), stack[stack_top].g00 + n, n * sizeof *(g00 + n));
				my_memcpy_gpu(stack[stack_top].g00 + n, stack[stack_top].g00, hn * sizeof *g00);
				my_memcpy_gpu(stack[stack_top].g00 + n + hn, stack[stack_top].g11, hn * sizeof *g00);

				/*
				 * The half-size Gram matrices for the recursive LDL tree
				 * building are now:
				 *   - left sub-tree: g00, g00+hn, g01
				 *   - right sub-tree: g11, g11+hn, g01+hn
				 * l10 is in tmp[].
				 */
				 
				/*
				 * We split t1 and use the first recursive call on the two
				 * halves, using the right sub-tree. The result is merged
				 * back into tmp + 2*n.
				 */
				stack[stack_top].is_z1 = 1;
				Zf(poly_split_fft_gpu)(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + 3 * n + hn, stack[stack_top].t0 + n, stack[stack_top].logn, gm_tab_sh);
				stack[stack_top + 1].t0 = stack[stack_top].t0 + 3 * n;
				stack[stack_top + 1].g00 = stack[stack_top].g11;
				stack[stack_top + 1].g11 = stack[stack_top].g00 + n + hn;
				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
				stack[stack_top + 1].is_z0 = 0;
				stack[++stack_top].is_z1 = 0;
			}
			else if (!stack[stack_top].is_z0)
			{
				/*
				 * Compute tb0 = t0 + (t1 - z1) * l10.
				 * At that point, l10 is in tmp, t1 is unmodified, and z1 is
				 * in tmp + (n << 1). The buffer in z1 is free.
				 *
				 * In the end, z1 is written over t1, and tb0 is in t0.
				 */
				my_memcpy_gpu(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + n, n * sizeof *(t0 + n));
				Zf(poly_sub_gpu)(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + (n << 2), stack[stack_top].logn);
				my_memcpy_gpu(stack[stack_top].t0 + n, stack[stack_top].t0 + (n << 2), n * sizeof *(t0 + (n << 1)));
				Zf(poly_mul_fft_gpu)(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + 3 * n, stack[stack_top].logn);
				Zf(poly_add_gpu)(stack[stack_top].t0, stack[stack_top].t0 + (n << 1), stack[stack_top].logn);
				/*
				 * Second recursive invocation, on the split tb0 (currently in t0)
				 * and the left sub-tree.
				 */
				stack[stack_top].is_z0 = 1;
				Zf(poly_split_fft_gpu)(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].t0, stack[stack_top].logn, gm_tab_sh);

				stack[stack_top + 1].t0 = stack[stack_top].t0 + (n << 1);
				stack[stack_top + 1].g00 = stack[stack_top].g00;
				stack[stack_top + 1].g11 = stack[stack_top].g00 + n;
				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
				stack[stack_top + 1].is_z0 = 0;
				stack[++stack_top].is_z1 = 0;
				
			}
			else
			{
				if (stack[stack_top].logn == orig_logn)
				{
					return;
				}
				else
				{   	
					if (!stack[--stack_top].is_z0)
					{
						
						Zf(poly_merge_fft_gpu)(stack[stack_top].t0 + (n << 3), stack[stack_top].t0 + 6 * n, stack[stack_top].t0 + 7 * n, stack[stack_top].logn, gm_tab_sh);
					}
					else
					{
						Zf(poly_merge_fft_gpu)(stack[stack_top].t0, stack[stack_top].t0 + (n << 2), stack[stack_top].t0 + 5 * n, stack[stack_top].logn, gm_tab_sh);
					}
				}
			}
		}
	}
}



__device__ static void
ffSampling_fft_gpu(samplerZ samp, void *samp_ctx,
    fpr *z0, fpr *z1,
    const fpr *tree,
    const fpr *t0, const fpr *t1, unsigned logn,
    fpr *tmp, const fpr *gm_tab_sh)
{
    typedef struct {
        unsigned logn;
        fpr *z0; fpr *z1;
        const fpr *t0; const fpr *t1;
        const fpr *tree;
        fpr *tmp;
        int state;
    } Frame;

    const int MAX_FRAMES = 32;
    Frame stack[MAX_FRAMES];
    int sp = 0;

    stack[sp++] = (Frame){ .logn = logn,
                           .z0 = z0, .z1 = z1,
                           .t0 = t0, .t1 = t1,
                           .tree = tree,
                           .tmp = tmp,
                           .state = 0 };

    while (sp > 0) {
        Frame fr = stack[--sp];

        if (fr.logn == 2) {
            fpr x0, x1, y0, y1, w0, w1, w2, w3, sigma;
            fpr a_re, a_im, b_re, b_im, c_re, c_im;
            const fpr *tree0 = fr.tree + 4;
            const fpr *tree1 = fr.tree + 8;

            a_re = fr.t1[0];
            a_im = fr.t1[2];
            b_re = fr.t1[1];
            b_im = fr.t1[3];
            c_re = fpr_add_gpu(a_re, b_re);
            c_im = fpr_add_gpu(a_im, b_im);
            w0 = fpr_half_gpu(c_re);
            w1 = fpr_half_gpu(c_im);
            c_re = fpr_sub_gpu(a_re, b_re);
            c_im = fpr_sub_gpu(a_im, b_im);
            w2 = fpr_mul_gpu(fpr_add_gpu(c_re, c_im), fpr_invsqrt8);
            w3 = fpr_mul_gpu(fpr_sub_gpu(c_im, c_re), fpr_invsqrt8);

            x0 = w2;
            x1 = w3;
            sigma = tree1[3];
            w2 = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            w3 = fpr_of_gpu(samp(samp_ctx, x1, sigma));
            a_re = fpr_sub_gpu(x0, w2);
            a_im = fpr_sub_gpu(x1, w3);
            b_re = tree1[0];
            b_im = tree1[1];
            c_re = fpr_sub_gpu(fpr_mul_gpu(a_re, b_re), fpr_mul_gpu(a_im, b_im));
            c_im = fpr_add_gpu(fpr_mul_gpu(a_re, b_im), fpr_mul_gpu(a_im, b_re));
            x0 = fpr_add_gpu(c_re, w0);
            x1 = fpr_add_gpu(c_im, w1);
            sigma = tree1[2];
            w0 = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            w1 = fpr_of_gpu(samp(samp_ctx, x1, sigma));

            a_re = w0;
            a_im = w1;
            b_re = w2;
            b_im = w3;
            c_re = fpr_mul_gpu(fpr_sub_gpu(b_re, b_im), fpr_invsqrt2);
            c_im = fpr_mul_gpu(fpr_add_gpu(b_re, b_im), fpr_invsqrt2);
            fr.z1[0] = w0 = fpr_add_gpu(a_re, c_re);
            fr.z1[2] = w2 = fpr_add_gpu(a_im, c_im);
            fr.z1[1] = w1 = fpr_sub_gpu(a_re, c_re);
            fr.z1[3] = w3 = fpr_sub_gpu(a_im, c_im);

            w0 = fpr_sub_gpu(fr.t1[0], w0);
            w1 = fpr_sub_gpu(fr.t1[1], w1);
            w2 = fpr_sub_gpu(fr.t1[2], w2);
            w3 = fpr_sub_gpu(fr.t1[3], w3);

            a_re = w0;
            a_im = w2;
            b_re = fr.tree[0];
            b_im = fr.tree[2];
            w0 = fpr_sub_gpu(fpr_mul_gpu(a_re, b_re), fpr_mul_gpu(a_im, b_im));
            w2 = fpr_add_gpu(fpr_mul_gpu(a_re, b_im), fpr_mul_gpu(a_im, b_re));
            a_re = w1;
            a_im = w3;
            b_re = fr.tree[1];
            b_im = fr.tree[3];
            w1 = fpr_sub_gpu(fpr_mul_gpu(a_re, b_re), fpr_mul_gpu(a_im, b_im));
            w3 = fpr_add_gpu(fpr_mul_gpu(a_re, b_im), fpr_mul_gpu(a_im, b_re));

            w0 = fpr_add_gpu(w0, fr.t0[0]);
            w1 = fpr_add_gpu(w1, fr.t0[1]);
            w2 = fpr_add_gpu(w2, fr.t0[2]);
            w3 = fpr_add_gpu(w3, fr.t0[3]);

            a_re = w0;
            a_im = w2;
            b_re = w1;
            b_im = w3;
            c_re = fpr_add_gpu(a_re, b_re);
            c_im = fpr_add_gpu(a_im, b_im);
            w0 = fpr_half_gpu(c_re);
            w1 = fpr_half_gpu(c_im);
            c_re = fpr_sub_gpu(a_re, b_re);
            c_im = fpr_sub_gpu(a_im, b_im);
            w2 = fpr_mul_gpu(fpr_add_gpu(c_re, c_im), fpr_invsqrt8);
            w3 = fpr_mul_gpu(fpr_sub_gpu(c_im, c_re), fpr_invsqrt8);

            x0 = w2;
            x1 = w3;
            sigma = tree0[3];
            w2 = y0 = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            w3 = y1 = fpr_of_gpu(samp(samp_ctx, x1, sigma));
            a_re = fpr_sub_gpu(x0, y0);
            a_im = fpr_sub_gpu(x1, y1);
            b_re = tree0[0];
            b_im = tree0[1];
            c_re = fpr_sub_gpu(fpr_mul_gpu(a_re, b_re), fpr_mul_gpu(a_im, b_im));
            c_im = fpr_add_gpu(fpr_mul_gpu(a_re, b_im), fpr_mul_gpu(a_im, b_re));
            x0 = fpr_add_gpu(c_re, w0);
            x1 = fpr_add_gpu(c_im, w1);
            sigma = tree0[2];
            w0 = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            w1 = fpr_of_gpu(samp(samp_ctx, x1, sigma));

            a_re = w0;
            a_im = w1;
            b_re = w2;
            b_im = w3;
            c_re = fpr_mul_gpu(fpr_sub_gpu(b_re, b_im), fpr_invsqrt2);
            c_im = fpr_mul_gpu(fpr_add_gpu(b_re, b_im), fpr_invsqrt2);
            fr.z0[0] = fpr_add_gpu(a_re, c_re);
            fr.z0[2] = fpr_add_gpu(a_im, c_im);
            fr.z0[1] = fpr_sub_gpu(a_re, c_re);
            fr.z0[3] = fpr_sub_gpu(a_im, c_im);
            continue;
        }

        if (fr.logn == 1) {
            fpr x0, x1, y0, y1, sigma;
            fpr a_re, a_im, b_re, b_im, c_re, c_im;

            x0 = fr.t1[0];
            x1 = fr.t1[1];
            sigma = fr.tree[3];
            fr.z1[0] = y0 = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            fr.z1[1] = y1 = fpr_of_gpu(samp(samp_ctx, x1, sigma));
            a_re = fpr_sub_gpu(x0, y0);
            a_im = fpr_sub_gpu(x1, y1);
            b_re = fr.tree[0];
            b_im = fr.tree[1];
            c_re = fpr_sub_gpu(fpr_mul_gpu(a_re, b_re), fpr_mul_gpu(a_im, b_im));
            c_im = fpr_add_gpu(fpr_mul_gpu(a_re, b_im), fpr_mul_gpu(a_im, b_re));
            x0 = fpr_add_gpu(c_re, fr.t0[0]);
            x1 = fpr_add_gpu(c_im, fr.t0[1]);
            sigma = fr.tree[2];
            fr.z0[0] = fpr_of_gpu(samp(samp_ctx, x0, sigma));
            fr.z0[1] = fpr_of_gpu(samp(samp_ctx, x1, sigma));
            continue;
        }

        if (fr.state == 0) {
            unsigned n = (size_t)1 << fr.logn;
            unsigned hn = n >> 1;
            const fpr *tree1 = fr.tree + n + ffLDL_treesize_gpu(fr.logn - 1);

            Zf(poly_split_fft_gpu)(fr.z1, fr.z1 + hn, fr.t1, fr.logn, gm_tab_sh);

            if (sp + 2 > MAX_FRAMES) {
                return;
            }
            Frame cont = fr;
            cont.state = 1;
            stack[sp++] = cont;

            Frame child;
            child.logn = fr.logn - 1;
            child.z0 = fr.tmp; 
            child.z1 = fr.tmp + hn;
            child.t0 = fr.z1;
            child.t1 = fr.z1 + hn;
            child.tree = tree1;
            child.tmp = fr.tmp + n;
            child.state = 0;
            stack[sp++] = child;

            continue;
        }

        if (fr.state == 1) {
            unsigned n = (size_t)1 << fr.logn;
            unsigned hn = n >> 1;
            const fpr *tree0 = fr.tree + n;
            Zf(poly_merge_fft_gpu)(fr.z1, fr.tmp, fr.tmp + hn, fr.logn, gm_tab_sh);

            my_memcpy_gpu(fr.tmp, fr.t1, n * sizeof *fr.t1);
            Zf(poly_sub_gpu)(fr.tmp, fr.z1, fr.logn);
            Zf(poly_mul_fft_gpu)(fr.tmp, fr.tree, fr.logn);
            Zf(poly_add_gpu)(fr.tmp, fr.t0, fr.logn);

            Zf(poly_split_fft_gpu)(fr.z0, fr.z0 + hn, fr.tmp, fr.logn, gm_tab_sh);

            if (sp + 2 > MAX_FRAMES) {
                return;
            }
            Frame cont = fr;
            cont.state = 2;
            stack[sp++] = cont;

            Frame child;
            child.logn = fr.logn - 1;
            child.z0 = fr.tmp;            
            child.z1 = fr.tmp + hn;
            child.t0 = fr.z0;             
            child.t1 = fr.z0 + hn;
            child.tree = tree0;
            child.tmp = fr.tmp + n;   
            child.state = 0;
            stack[sp++] = child;

            continue;
        }

        if (fr.state == 2) {
            unsigned n = (size_t)1 << fr.logn;
            unsigned hn = n >> 1;
            Zf(poly_merge_fft_gpu)(fr.z0, fr.tmp, fr.tmp + hn, fr.logn,gm_tab_sh);
            continue;
        }

    } 
}




/*
 * Compute a signature: the signature contains two vectors, s1 and s2.
 * The s1 vector is not returned. The squared norm of (s1,s2) is
 * computed, and if it is short enough, then s2 is returned into the
 * s2[] buffer, and 1 is returned; otherwise, s2[] is untouched and 0 is
 * returned; the caller should then try again. This function uses an
 * expanded key.
 *
 * tmp[] must have room for at least six polynomials.
 */
__device__ static int
do_sign_tree_gpu(samplerZ samp, void *samp_ctx, int16_t *s2,
	const fpr *expanded_key,
	const uint16_t *hm,
	unsigned logn, fpr *tmp, const fpr *gm_tab_sh)
{
	size_t n, u;
	fpr *t0, *t1, *tx, *ty;
	const fpr *b00, *b01, *b10, *b11, *tree;
	fpr ni;
	uint32_t sqn, ng;
	int16_t *s1tmp, *s2tmp;

	n = MKN(logn);
	t0 = tmp;
	t1 = t0 + n;
	b00 = expanded_key + skoff_b00_gpu(logn);
	b01 = expanded_key + skoff_b01_gpu(logn);
	b10 = expanded_key + skoff_b10_gpu(logn);
	b11 = expanded_key + skoff_b11_gpu(logn);
	tree = expanded_key + skoff_tree_gpu(logn);

	/*
	 * Set the target vector to [hm, 0] (hm is the hashed message).
	 */
	for (u = 0; u < n; u ++) {
		t0[u] = fpr_of_gpu(hm[u]);
		/* This is implicit.
		t1[u] = fpr_zero;
		*/
	}

	/*
	 * Apply the lattice basis to obtain the real target
	 * vector (after normalization with regards to modulus).
	 */
	Zf(FFT_gpu)(t0, logn, gm_tab_sh);
	ni = fpr_inverse_of_q;
	my_memcpy_gpu(t1, t0, n * sizeof *t0);
	Zf(poly_mul_fft_gpu)(t1, b01, logn);
	Zf(poly_mulconst_gpu)(t1, fpr_neg_gpu(ni), logn);
	Zf(poly_mul_fft_gpu)(t0, b11, logn);
	Zf(poly_mulconst_gpu)(t0, ni, logn);

	tx = t1 + n;
	ty = tx + n;

	/*
	 * Apply sampling. Output is written back in [tx, ty].
	 */
	ffSampling_fft_gpu(samp, samp_ctx, tx, ty, tree, t0, t1, logn, ty + n, gm_tab_sh);

	/*
	 * Get the lattice point corresponding to that tiny vector.
	 */
	my_memcpy_gpu(t0, tx, n * sizeof *tx);
	my_memcpy_gpu(t1, ty, n * sizeof *ty);
	Zf(poly_mul_fft_gpu)(tx, b00, logn);
	Zf(poly_mul_fft_gpu)(ty, b10, logn);
	Zf(poly_add_gpu)(tx, ty, logn);
	my_memcpy_gpu(ty, t0, n * sizeof *t0);
	Zf(poly_mul_fft_gpu)(ty, b01, logn);

	my_memcpy_gpu(t0, tx, n * sizeof *tx);
	Zf(poly_mul_fft_gpu)(t1, b11, logn);
	Zf(poly_add_gpu)(t1, ty, logn);

	Zf(iFFT_gpu)(t0, logn, gm_tab_sh);
	Zf(iFFT_gpu)(t1, logn, gm_tab_sh);

	/*
	 * Compute the signature.
	 */
	s1tmp = (int16_t *)tx;
	sqn = 0;
	ng = 0;
	for (u = 0; u < n; u ++) {
		int32_t z;

		z = (int32_t)hm[u] - (int32_t)fpr_rint_gpu(t0[u]);
		sqn += (uint32_t)(z * z);
		ng |= sqn;
		s1tmp[u] = (int16_t)z;
	}
	sqn |= -(ng >> 31);

	/*
	 * With "normal" degrees (e.g. 512 or 1024), it is very
	 * improbable that the computed vector is not short enough;
	 * however, it may happen in practice for the very reduced
	 * versions (e.g. degree 16 or below). In that case, the caller
	 * will loop, and we must not write anything into s2[] because
	 * s2[] may overlap with the hashed message hm[] and we need
	 * hm[] for the next iteration.
	 */
	s2tmp = (int16_t *)tmp;
	for (u = 0; u < n; u ++) {
		s2tmp[u] = (int16_t)-fpr_rint_gpu(t1[u]);
	}
	if (Zf(is_short_half_gpu)(sqn, s2tmp, logn)) {
		my_memcpy_gpu(s2, s2tmp, n * sizeof *s2);
		my_memcpy_gpu(tmp, s1tmp, n * sizeof *s1tmp);
		return 1;
	}
	return 0;
}

/*
 * Compute a signature: the signature contains two vectors, s1 and s2.
 * The s1 vector is not returned. The squared norm of (s1,s2) is
 * computed, and if it is short enough, then s2 is returned into the
 * s2[] buffer, and 1 is returned; otherwise, s2[] is untouched and 0 is
 * returned; the caller should then try again.
 *
 * tmp[] must have room for at least nine polynomials.
 */
__device__ static int
do_sign_dyn_gpu(samplerZ samp, void *samp_ctx, int16_t *s2,
	const int8_t *f, const int8_t *g,
	const int8_t *F, const int8_t *G,
	const uint16_t *hm, unsigned logn, fpr *tmp, const fpr *gm_tab_sh)
{
	size_t n, u;
	fpr *t0, *t1, *tx, *ty;
	fpr *b00, *b01, *b10, *b11, *g00, *g01, *g11;
	fpr ni;
	uint32_t sqn, ng;
	int16_t *s1tmp, *s2tmp;


	n = MKN(logn);
	/*
	 * Lattice basis is B = [[g, -f], [G, -F]]. We convert it to FFT.
	 */
	b00 = tmp;
	b01 = b00 + n;
	b10 = b01 + n;
	b11 = b10 + n;
	smallints_to_fpr_gpu(b01, f, logn);
	smallints_to_fpr_gpu(b00, g, logn);
	smallints_to_fpr_gpu(b11, F, logn);
	smallints_to_fpr_gpu(b10, G, logn);
	Zf(FFT_gpu)(b01, logn, gm_tab_sh);
	Zf(FFT_gpu)(b00, logn, gm_tab_sh);
	Zf(FFT_gpu)(b11, logn, gm_tab_sh);
	Zf(FFT_gpu)(b10, logn, gm_tab_sh);
	Zf(poly_neg_gpu)(b01, logn);
	Zf(poly_neg_gpu)(b11, logn);

	/*
	 * Compute the Gram matrix G = B·B*. Formulas are:
	 *   g00 = b00*adj(b00) + b01*adj(b01)
	 *   g01 = b00*adj(b10) + b01*adj(b11)
	 *   g10 = b10*adj(b00) + b11*adj(b01)
	 *   g11 = b10*adj(b10) + b11*adj(b11)
	 *
	 * For historical reasons, this implementation uses
	 * g00, g01 and g11 (upper triangle). g10 is not kept
	 * since it is equal to adj(g01).
	 *
	 * We _replace_ the matrix B with the Gram matrix, but we
	 * must keep b01 and b11 for computing the target vector.
	 */
	t0 = b11 + n;//tmp+2048
	t1 = t0 + n;//tmp+2560
	my_memcpy_gpu(t0, b01, n * sizeof *b01);
	Zf(poly_mulselfadj_fft_gpu)(t0, logn);  
	my_memcpy_gpu(t1, b00, n * sizeof *b00);
	Zf(poly_muladj_fft_gpu)(t1, b10, logn); 
	Zf(poly_mulselfadj_fft_gpu)(b00, logn); 
	Zf(poly_add_gpu)(b00, t0, logn);      
	my_memcpy_gpu(t0, b01, n * sizeof *b01);
	Zf(poly_muladj_fft_gpu)(b01, b11, logn);
	Zf(poly_add_gpu)(b01, t1, logn);      

	Zf(poly_mulselfadj_fft_gpu)(b10, logn); 
	my_memcpy_gpu(t1, b11, n * sizeof *b11);
	Zf(poly_mulselfadj_fft_gpu)(t1, logn);  
	Zf(poly_add_gpu)(b10, t1, logn);      

	/*
	 * We rename variables to make things clearer. The three elements
	 * of the Gram matrix uses the first 3*n slots of tmp[], followed
	 * by b11 and b01 (in that order).
	 */
	g00 = b00;
	g01 = b01;
	g11 = b10;
	b01 = t0;
	t0 = b01 + n;
	t1 = t0 + n;

	/*
	 * Memory layout at that point:
	 *   g00 g01 g11 b11 b01 t0 t1
	 */

	/*
	 * Set the target vector to [hm, 0] (hm is the hashed message).
	 */
	for (u = 0; u < n; u ++) {
		t0[u] = fpr_of_gpu(hm[u]);
		/* This is implicit.
		t1[u] = fpr_zero;
		*/ 
	}

	/*
	 * Apply the lattice basis to obtain the real target
	 * vector (after normalization with regards to modulus).
	 */
	Zf(FFT_gpu)(t0, logn, gm_tab_sh);
	ni = fpr_inverse_of_q;
	my_memcpy_gpu(t1, t0, n * sizeof *t0);
	Zf(poly_mul_fft_gpu)(t1, b01, logn);
	Zf(poly_mulconst_gpu)(t1, fpr_neg_gpu(ni), logn);
	Zf(poly_mul_fft_gpu)(t0, b11, logn);
	Zf(poly_mulconst_gpu)(t0, ni, logn);

	/*
	 * b01 and b11 can be discarded, so we move back (t0,t1).
	 * Memory layout is now:
	 *      g00 g01 g11 t0 t1
	 */
	my_memcpy_gpu(b11, t0, n * 2 * sizeof *t0);
	t0 = g11 + n;
	t1 = t0 + n;
	/*
	 * Apply sampling; result is written over (t0,t1).
	 */
	ffSampling_fft_dyntree_gpu(samp, samp_ctx,
		t0, t1, g00, g01, g11, logn, logn, t1 + n, gm_tab_sh);
	/*
	 * We arrange the layout back to:
	 *     b00 b01 b10 b11 t0 t1
	 *
	 * We did not conserve the matrix basis, so we must recompute
	 * it now.
	 */
	b00 = tmp;
	b01 = b00 + n;
	b10 = b01 + n;
	b11 = b10 + n;
	my_memmove_gpu(b11 + n, t0, n * 2 * sizeof *t0);
	t0 = b11 + n;
	t1 = t0 + n;
	smallints_to_fpr_gpu(b01, f, logn);
	smallints_to_fpr_gpu(b00, g, logn);
	smallints_to_fpr_gpu(b11, F, logn);
	smallints_to_fpr_gpu(b10, G, logn);
	Zf(FFT_gpu)(b01, logn, gm_tab_sh);
	Zf(FFT_gpu)(b00, logn, gm_tab_sh);
	Zf(FFT_gpu)(b11, logn, gm_tab_sh);
	Zf(FFT_gpu)(b10, logn, gm_tab_sh);
	Zf(poly_neg_gpu)(b01, logn);
	Zf(poly_neg_gpu)(b11, logn);
	tx = t1 + n;
	ty = tx + n;

	/*
	 * Get the lattice point corresponding to that tiny vector.
	 */
	my_memcpy_gpu(tx, t0, n * sizeof *t0);
	my_memcpy_gpu(ty, t1, n * sizeof *t1);
	Zf(poly_mul_fft_gpu)(tx, b00, logn);
	Zf(poly_mul_fft_gpu)(ty, b10, logn);
	Zf(poly_add_gpu)(tx, ty, logn);
	my_memcpy_gpu(ty, t0, n * sizeof *t0);
	Zf(poly_mul_fft_gpu)(ty, b01, logn);

	my_memcpy_gpu(t0, tx, n * sizeof *tx);
	Zf(poly_mul_fft_gpu)(t1, b11, logn);
	Zf(poly_add_gpu)(t1, ty, logn);
	Zf(iFFT_gpu)(t0, logn, gm_tab_sh);
	Zf(iFFT_gpu)(t1, logn, gm_tab_sh);

	s1tmp = (int16_t *)tx;
	sqn = 0;
	ng = 0;
	for (u = 0; u < n; u ++) {
		int32_t z;

		z = (int32_t)hm[u] - (int32_t)fpr_rint_gpu(t0[u]);
		sqn += (uint32_t)(z * z);
		ng |= sqn;
		s1tmp[u] = (int16_t)z;
	}
	sqn |= -(ng >> 31);

	/*
	 * With "normal" degrees (e.g. 512 or 1024), it is very
	 * improbable that the computed vector is not short enough;
	 * however, it may happen in practice for the very reduced
	 * versions (e.g. degree 16 or below). In that case, the caller
	 * will loop, and we must not write anything into s2[] because
	 * s2[] may overlap with the hashed message hm[] and we need
	 * hm[] for the next iteration.
	 */
	s2tmp = (int16_t *)tmp;
	for (u = 0; u < n; u ++) {
		s2tmp[u] = (int16_t)-fpr_rint_gpu(t1[u]);
	}
	if (Zf(is_short_half_gpu)(sqn, s2tmp, logn)) {
		my_memcpy_gpu(s2, s2tmp, n * sizeof *s2);
		my_memcpy_gpu(tmp, s1tmp, n * sizeof *s1tmp);
		return 1;
	}
	return 0;
}

/*
 * Sample an integer value along a half-gaussian distribution centered
 * on zero and standard deviation 1.8205, with a precision of 72 bits.
 */
__device__ int
Zf(gaussian0_sampler_gpu)(prng *p)
{

	static const uint32_t dist[] = {
		10745844u,  3068844u,  3741698u,
		 5559083u,  1580863u,  8248194u,
		 2260429u, 13669192u,  2736639u,
		  708981u,  4421575u, 10046180u,
		  169348u,  7122675u,  4136815u,
		   30538u, 13063405u,  7650655u,
		    4132u, 14505003u,  7826148u,
		     417u, 16768101u, 11363290u,
		      31u,  8444042u,  8086568u,
		       1u, 12844466u,   265321u,
		       0u,  1232676u, 13644283u,
		       0u,    38047u,  9111839u,
		       0u,      870u,  6138264u,
		       0u,       14u, 12545723u,
		       0u,        0u,  3104126u,
		       0u,        0u,    28824u,
		       0u,        0u,      198u,
		       0u,        0u,        1u
	};

	uint32_t v0, v1, v2, hi;
	uint64_t lo;
	size_t u;
	int z;

	/*
	 * Get a random 72-bit value, into three 24-bit limbs v0..v2.
	 */
	lo = prng_get_u64_gpu(p);
	hi = prng_get_u8_gpu(p);
	v0 = (uint32_t)lo & 0xFFFFFF;
	v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
	v2 = (uint32_t)(lo >> 48) | (hi << 16);

	/*
	 * Sampled value is z, such that v0..v2 is lower than the first
	 * z elements of the table.
	 */
	z = 0;
	for (u = 0; u < (sizeof dist) / sizeof(dist[0]); u += 3) {
		uint32_t w0, w1, w2, cc;

		w0 = dist[u + 2];
		w1 = dist[u + 1];
		w2 = dist[u + 0];
		cc = (v0 - w0) >> 31;
		cc = (v1 - w1 - cc) >> 31;
		cc = (v2 - w2 - cc) >> 31;
		z += (int)cc;
	}
	return z;

}

/*
 * Sample a bit with probability exp(-x) for some x >= 0.
 */
__device__ static int
BerExp_gpu(prng *p, fpr x, fpr ccs)
{
	int s, i;
	fpr r;
	uint32_t sw, w;
	uint64_t z;

	/*
	 * Reduce x modulo log(2): x = s*log(2) + r, with s an integer,
	 * and 0 <= r < log(2). Since x >= 0, we can use fpr_trunc_gpu().
	 */
	s = (int)fpr_trunc_gpu(fpr_mul_gpu(x, fpr_inv_log2));
	r = fpr_sub_gpu(x, fpr_mul_gpu(fpr_of_gpu(s), fpr_log2));

	/*
	 * It may happen (quite rarely) that s >= 64; if sigma = 1.2
	 * (the minimum value for sigma), r = 0 and b = 1, then we get
	 * s >= 64 if the half-Gaussian produced a z >= 13, which happens
	 * with probability about 0.000000000230383991, which is
	 * approximatively equal to 2^(-32). In any case, if s >= 64,
	 * then BerExp_gpu will be non-zero with probability less than
	 * 2^(-64), so we can simply saturate s at 63.
	 */
	sw = (uint32_t)s;
	sw ^= (sw ^ 63) & -((63 - sw) >> 31);
	s = (int)sw;

	/*
	 * Compute exp(-r); we know that 0 <= r < log(2) at this point, so
	 * we can use fpr_expm_p63_gpu(), which yields a result scaled to 2^63.
	 * We scale it up to 2^64, then right-shift it by s bits because
	 * we really want exp(-x) = 2^(-s)*exp(-r).
	 *
	 * The "-1" operation makes sure that the value fits on 64 bits
	 * (i.e. if r = 0, we may get 2^64, and we prefer 2^64-1 in that
	 * case). The bias is negligible since fpr_expm_p63_gpu() only computes
	 * with 51 bits of precision or so.
	 */
	z = ((fpr_expm_p63_gpu(r, ccs) << 1) - 1) >> s;

	/*
	 * Sample a bit with probability exp(-x). Since x = s*log(2) + r,
	 * exp(-x) = 2^-s * exp(-r), we compare lazily exp(-x) with the
	 * PRNG output to limit its consumption, the sign of the difference
	 * yields the expected result.
	 */
	i = 64;
	do {
		i -= 8;
		w = prng_get_u8_gpu(p) - ((uint32_t)(z >> i) & 0xFF);
	} while (!w && i > 0);
	return (int)(w >> 31);
}

/*
 * The sampler produces a random integer that follows a discrete Gaussian
 * distribution, centered on mu, and with standard deviation sigma. The
 * provided parameter isigma is equal to 1/sigma.
 *
 * The value of sigma MUST lie between 1 and 2 (i.e. isigma lies between
 * 0.5 and 1); in Falcon, sigma should always be between 1.2 and 1.9.
 */
__device__ int
Zf(sampler_gpu)(void *ctx, fpr mu, fpr isigma)
{
	sampler_context *spc;
	int s;
	fpr r, dss, ccs;

	spc = (sampler_context*)ctx;

	/*
	 * Center is mu. We compute mu = s + r where s is an integer
	 * and 0 <= r < 1.
	 */
	s = (int)fpr_floor_gpu(mu);
	r = fpr_sub_gpu(mu, fpr_of_gpu(s));

	/*
	 * dss = 1/(2*sigma^2) = 0.5*(isigma^2).
	 */	
	dss = fpr_half_gpu(fpr_sqr_gpu(isigma));

	/*
	 * ccs = sigma_min / sigma = sigma_min * isigma.
	 */
	ccs = fpr_mul_gpu(isigma, spc->sigma_min);

	/*
	 * We now need to sample on center r.
	 */
	for (;;) {
		int z0, z, b;
		fpr x;

		/*
		 * Sample z for a Gaussian distribution. Then get a
		 * random bit b to turn the sampling into a bimodal
		 * distribution: if b = 1, we use z+1, otherwise we
		 * use -z. We thus have two situations:
		 *
		 *  - b = 1: z >= 1 and sampled against a Gaussian
		 *    centered on 1.
		 *  - b = 0: z <= 0 and sampled against a Gaussian
		 *    centered on 0.
		 */
		z0 = Zf(gaussian0_sampler_gpu)(&spc->p);
		b = (int)prng_get_u8_gpu(&spc->p) & 1;
		z = b + ((b << 1) - 1) * z0;

		/*
		 * Rejection sampling. We want a Gaussian centered on r;
		 * but we sampled against a Gaussian centered on b (0 or
		 * 1). But we know that z is always in the range where
		 * our sampling distribution is greater than the Gaussian
		 * distribution, so rejection works.
		 *
		 * We got z with distribution:
		 *    G(z) = exp(-((z-b)^2)/(2*sigma0^2))
		 * We target distribution:
		 *    S_gpu(z) = exp(-((z-r)^2)/(2*sigma^2))
		 * Rejection sampling works by keeping the value z with
		 * probability S_gpu(z)/G(z), and starting again otherwise.
		 * This requires S_gpu(z) <= G(z), which is the case here.
		 * Thus, we simply need to keep our z with probability:
		 *    P = exp(-x)
		 * where:
		 *    x = ((z-r)^2)/(2*sigma^2) - ((z-b)^2)/(2*sigma0^2)
		 *
		 * Here, we scale up the Bernouilli distribution, which
		 * makes rejection more probable, but makes rejection
		 * rate sufficiently decorrelated from the Gaussian
		 * center and standard deviation that the whole sampler
		 * can be said to be constant-time.
		 */	
		x = fpr_mul_gpu(fpr_sqr_gpu(fpr_sub_gpu(fpr_of_gpu(z), r)), dss);
		x = fpr_sub_gpu(x, fpr_mul_gpu(fpr_of_gpu(z0 * z0), fpr_inv_2sqrsigma0));

		if (BerExp_gpu(&spc->p, x, ccs)) {
			/*
			 * Rejection sampling was centered on r, but the
			 * actual center is mu = s + r.
			 */
			return s + z;
		}
	}
}

/* see inner.h */
__device__ void
Zf(sign_tree_gpu)(int16_t *sig, inner_shake256_context *rng,
	const fpr *expanded_key,
	const uint16_t *hm, unsigned logn, uint8_t *tmp, const fpr *gm_tab_sh)
{
	fpr *ftmp;

	ftmp = (fpr *)tmp;
	for (;;) {
		/*
		 * Signature produces short vectors s1 and s2. The
		 * signature is acceptable only if the aggregate vector
		 * s1,s2 is short; we must use the same bound as the
		 * verifier.
		 *
		 * If the signature is acceptable, then we return only s2
		 * (the verifier recomputes s1 from s2, the hashed message,
		 * and the public key).
		 */
		sampler_context spc;
		samplerZ samp;
		void *samp_ctx;

		/*
		 * Normal sampling. We use a fast PRNG seeded from our
		 * SHAKE context ('rng').
		 */
		spc.sigma_min = fpr_sigma_min_gpu[logn];
		Zf(prng_init_gpu)(&spc.p, rng);
		samp = Zf(sampler_gpu);
		samp_ctx = &spc;

		/*
		 * Do the actual signature.
		 */
		if (do_sign_tree_gpu(samp, samp_ctx, sig,
			expanded_key, hm, logn, ftmp, gm_tab_sh))
		{
			break;
		}
	}
}

/* see inner.h */
__device__ void
Zf(sign_dyn_gpu)(int16_t *sig, inner_shake256_context *rng,
	const int8_t *f, const int8_t *g,
	const int8_t *F, const int8_t *G,
	const uint16_t *hm, unsigned logn, uint8_t *tmp, const fpr *gm_tab_sh)
{
	fpr *ftmp;

	ftmp = (fpr *)tmp;
	for (;;) {
		/*
		 * Signature produces short vectors s1 and s2. The
		 * signature is acceptable only if the aggregate vector
		 * s1,s2 is short; we must use the same bound as the
		 * verifier.
		 *
		 * If the signature is acceptable, then we return only s2
		 * (the verifier recomputes s1 from s2, the hashed message,
		 * and the public key).
		 */
		sampler_context spc;
		samplerZ samp;
		void *samp_ctx;

		/*
		 * Normal sampling. We use a fast PRNG seeded from our
		 * SHAKE context ('rng').
		 */
		spc.sigma_min = fpr_sigma_min_gpu[logn];
		Zf(prng_init_gpu)(&spc.p, rng);
		samp = Zf(sampler_gpu);
		samp_ctx = &spc;

		/*
		 * Do the actual signature.
		 */
		if (do_sign_dyn_gpu(samp, samp_ctx, sig,
			f, g, F, G, hm, logn, ftmp, gm_tab_sh))
		{
			break;
		}
	}
}
