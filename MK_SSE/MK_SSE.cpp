#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <iostream>
#include "time.h" 
#include <string>
#include <vector>
#include <intrin.h>

#define THREADS 4

using namespace std;

vector<vector<float>> FillArr();
__m128i allRandom(__m128i &seed);
__m128 RndKoeff(__m128i &seed);
float g(int a, int b);
float MK();

int an = 10, bn = 10, inn = 5, jnn = 5;
long int N = 1000000;

vector<vector<float>> ar = FillArr();

int main()
{
	float beg = clock();
	float sol = MK();
	printf("Solution at point [%d %d] = %f\n", inn, jnn, sol);
	cout << "time = " << (clock() - beg) / 1000.0 << endl;

	/*inn, jnn = 5, 4;
	sol = MK();
	printf("Solution at point [%d %d] = %f\n",inn, jnn, sol);

	for (int i = 1; i < an - 1; i++) {
		for (int j = 1; j < bn - 1; j++) {
			inn, jnn = i, j;
			sol = MK();
			cout << sol << "  ";
		}
		cout << endl;
	}*/

	return 0;
}

__m128i allRandom(__m128i &seed) {
	__m128i r = _mm_set1_epi32(3);
	__m128i nm1 = _mm_set1_epi32(825); // int nm1 = 825
	__m128i nm2 = _mm_set1_epi32(23964); // int nm2 = 23964
	seed = _mm_add_epi32(_mm_mul_epi32(nm1, seed), nm2); // seed = nm1*seed + nm2

	__m128i nm3 = _mm_set1_epi32(328);
	__m128 nm4 = _mm_set1_ps(327);
	__m128i result = _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_mul_epi32(seed, nm3)), nm4));
	result = _mm_and_si128(result, r); // result = result % 3
	return result; 
}

__m128 RndKoeff(__m128i &seed) {
	__m128i a = _mm_set1_epi32(inn);
	__m128i b = _mm_set1_epi32(jnn);

	__m128 ret = _mm_setzero_ps();

	__m128i zero = _mm_set1_epi32(0);
	__m128i one = _mm_set1_epi32(1);
	__m128i two = _mm_set1_epi32(2);
	__m128i three = _mm_set1_epi32(3);

	__m128i bord = _mm_set1_epi32(1);

	__m128i lx = _mm_set1_epi32(0);
	__m128i rx = _mm_set1_epi32(an-1);
	__m128i ty = _mm_set1_epi32(0);
	__m128i by = _mm_set1_epi32(bn-1);

	__m128i dx = _mm_set1_epi32(1);
	__m128i dy = _mm_set1_epi32(1);

	while (_mm_test_all_zeros(bord, one) != 1) {
		__m128i rnd = allRandom(seed);

		// смещение по x
		__m128i xmask1 = _mm_cmpeq_epi32(rnd, zero);
		__m128i xshift1 = _mm_and_si128(xmask1, dx);
		xshift1 = _mm_and_si128(xshift1, bord); // достигло ли границы
		a = _mm_subs_epi16(a, xshift1);

		__m128i xmask2 = _mm_cmpeq_epi32(rnd, one);
		__m128i xshift2 = _mm_and_si128(xmask2, dx);
		xshift2 = _mm_and_si128(xshift2, bord);
		a = _mm_add_epi32(a, xshift2);

		// смещение по y
		__m128i ymask1 = _mm_cmpeq_epi32(rnd, two);
		__m128i yshift1 = _mm_and_si128(ymask1, dy);
		yshift1 = _mm_and_si128(yshift1, bord);
		b = _mm_subs_epi16(b, yshift1);

		__m128i ymask2 = _mm_cmpeq_epi32(rnd, three);
		__m128i yshift2 = _mm_and_si128(ymask2, dy);
		yshift2 = _mm_and_si128(yshift2, bord);
		b = _mm_add_epi32(b, yshift2);

		__m128i bmask;

		bmask = _mm_cmpeq_epi32(a, lx);
		// Инверсия на множестве {0,1} (0->1, 1->0)
		__m128i nota1 = _mm_and_si128(_mm_add_epi32(bmask, one), one);
		bord = _mm_and_si128(nota1, bord);

		bmask = _mm_cmpeq_epi32(a, rx);
		bord = _mm_and_si128(_mm_and_si128(_mm_add_epi32(bmask, one), one), bord);

		bmask = _mm_cmpeq_epi32(b, by);
		bord = _mm_and_si128(_mm_and_si128(_mm_add_epi32(bmask, one), one), bord);

		bmask = _mm_cmpeq_epi32(b, ty);
		bord = _mm_and_si128(_mm_and_si128(_mm_add_epi32(bmask, one), one), bord);
	}

	ret = _mm_set_ps(ar[a.m128i_i32[0]][b.m128i_i32[0]], ar[a.m128i_i32[1]][b.m128i_i32[1]],
		ar[a.m128i_i32[2]][b.m128i_i32[2]], ar[a.m128i_i32[3]][b.m128i_i32[3]]);

	return ret;
}

float g(int a, int b) {
	return sin(a + 3.1415 / 2) * sin(3 * b + 3.1415 / 2);
}

vector<vector<float>> FillArr() {
	vector<vector<float>> ar(an, vector <float>(bn));
	for (int i = 0; i < an; i++) {
		for (int j = 0; j < bn; j++) {
			if (i != 0 && i != an - 1 && j != 0 && j != bn - 1) {
				ar[i][j] = 0;
			}
			else {
				ar[i][j] = g(i, j);
			}
		}
	}
	return ar;
}

float MK() {
	float ret = 0;

	__m128i seed = _mm_set_epi32(0, 1, 2, 3);
	__m128 x = _mm_setzero_ps();
	__m128 y;
	for (int i = 0; i + THREADS <= N; i += THREADS)
	{
		y = RndKoeff(seed);
		x = _mm_add_ps(x, y);
	}
	float *rt = (float*)&x;
	for (int i = 0; i < THREADS; i++)
	{
		ret += rt[i];
	}

	ret = (double)ret;
	ret = ret / (N);
	return ret;
}