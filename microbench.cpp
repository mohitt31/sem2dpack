#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#if defined(__AVX2__) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"
#endif

const int NGLL = 5;
const int NELEM = 12800;
const int W = 4;

// REFERENCE kernel for a single element
void elast_kd2_psv_ref_single(
    const double displ[NGLL][NGLL][2],
    const double a[NGLL][NGLL][6],
    const double H[NGLL][NGLL],
    const double Ht[NGLL][NGLL],
    double f[NGLL][NGLL][2]
) {
    double dUx_dxi[NGLL][NGLL] = {{0}};
    double dUz_dxi[NGLL][NGLL] = {{0}};
    double dUx_deta[NGLL][NGLL] = {{0}};
    double dUz_deta[NGLL][NGLL] = {{0}};

    for (int j = 0; j < NGLL; ++j) {
        for (int k = 0; k < NGLL; ++k) {
            for (int i = 0; i < NGLL; ++i) {
                dUx_dxi[i][j]  += Ht[i][k] * displ[k][j][0];
                dUz_dxi[i][j]  += Ht[i][k] * displ[k][j][1];
                dUx_deta[i][j] += displ[i][k][0] * H[k][j];
                dUz_deta[i][j] += displ[i][k][1] * H[k][j];
            }
        }
    }

    for (int i = 0; i < NGLL; ++i) {
        for (int j = 0; j < NGLL; ++j) {
            f[i][j][0] = 0.0;
            f[i][j][1] = 0.0;
        }
    }

    for (int j = 0; j < NGLL; ++j) {
        for (int k = 0; k < NGLL; ++k) {
            for (int i = 0; i < NGLL; ++i) {
                f[i][j][0] += H[i][k] * ( a[k][j][0]*dUx_dxi[k][j] + a[k][j][1]*dUz_deta[k][j] )
                            + a[i][k][3] * (dUx_deta[i][k] + dUz_dxi[i][k]) * Ht[k][j];
                               
                f[i][j][1] += H[i][k] * ( a[k][j][4]*dUx_deta[k][j] + a[k][j][5]*dUz_dxi[k][j] )
                            + ( a[i][k][1]*dUx_dxi[i][k] + a[i][k][2]*dUz_deta[i][k] ) * Ht[k][j];
            }
        }
    }
}

// Full reference kernel over all elements
void elast_kd2_psv_ref(
    const double displ[][NGLL][NGLL][2],
    const double a[][NGLL][NGLL][6],
    const double H[NGLL][NGLL],
    const double Ht[NGLL][NGLL],
    double f[][NGLL][NGLL][2]
) {
    for (int e = 0; e < NELEM; ++e) {
        elast_kd2_psv_ref_single(displ[e], a[e], H, Ht, f[e]);
    }
}

// Folded operator coefficients (5x5 centro-antisymmetric M). Indices 0..2 only.
struct EOfold { double p0[3], p1[3], c[3], m0[3], m1[3]; };

// LEFT apply y = M*x : fold COLUMNS of M
static EOfold fold_cols(const double M[5][5]){
    EOfold f;
    for(int i=0;i<3;++i){
        f.p0[i]=M[i][0]+M[i][4]; f.p1[i]=M[i][1]+M[i][3]; f.c[i]=M[i][2];
        f.m0[i]=M[i][0]-M[i][4]; f.m1[i]=M[i][1]-M[i][3];
    }
    return f;
}
// RIGHT apply y = x*M : fold ROWS of M
static EOfold fold_rows(const double M[5][5]){
    EOfold f;
    for(int j=0;j<3;++j){
        f.p0[j]=M[0][j]+M[4][j]; f.p1[j]=M[1][j]+M[3][j]; f.c[j]=M[2][j];
        f.m0[j]=M[0][j]-M[4][j]; f.m1[j]=M[1][j]-M[3][j];
    }
    return f;
}

// y = M * x  (each column of x)
static inline void eo_left(const EOfold& F, const __m256d x[5][5], __m256d y[5][5]){
    const __m256d half=_mm256_set1_pd(0.5);
    for(int j=0;j<5;++j){
        __m256d xs0=_mm256_add_pd(x[0][j],x[4][j]);
        __m256d xs1=_mm256_add_pd(x[1][j],x[3][j]);
        __m256d xc2=_mm256_add_pd(x[2][j],x[2][j]);      // 2*x2
        __m256d xa0=_mm256_sub_pd(x[0][j],x[4][j]);
        __m256d xa1=_mm256_sub_pd(x[1][j],x[3][j]);
        for(int i=0;i<3;++i){
            __m256d S=_mm256_mul_pd(_mm256_set1_pd(F.p0[i]),xs0);
            S=_mm256_fmadd_pd(_mm256_set1_pd(F.p1[i]),xs1,S);
            S=_mm256_fmadd_pd(_mm256_set1_pd(F.c[i]),xc2,S);
            __m256d A=_mm256_mul_pd(_mm256_set1_pd(F.m0[i]),xa0);
            A=_mm256_fmadd_pd(_mm256_set1_pd(F.m1[i]),xa1,A);
            y[i][j]=_mm256_mul_pd(_mm256_add_pd(A,S),half);
            if(i<2) y[4-i][j]=_mm256_mul_pd(_mm256_sub_pd(A,S),half);
        }
    }
}

// y = x * M  (each row of x)
static inline void eo_right(const EOfold& F, const __m256d x[5][5], __m256d y[5][5]){
    const __m256d half=_mm256_set1_pd(0.5);
    for(int i=0;i<5;++i){
        __m256d us0=_mm256_add_pd(x[i][0],x[i][4]);
        __m256d us1=_mm256_add_pd(x[i][1],x[i][3]);
        __m256d uc2=_mm256_add_pd(x[i][2],x[i][2]);
        __m256d ua0=_mm256_sub_pd(x[i][0],x[i][4]);
        __m256d ua1=_mm256_sub_pd(x[i][1],x[i][3]);
        for(int j=0;j<3;++j){
            __m256d S=_mm256_mul_pd(_mm256_set1_pd(F.p0[j]),us0);
            S=_mm256_fmadd_pd(_mm256_set1_pd(F.p1[j]),us1,S);
            S=_mm256_fmadd_pd(_mm256_set1_pd(F.c[j]),uc2,S);
            __m256d A=_mm256_mul_pd(_mm256_set1_pd(F.m0[j]),ua0);
            A=_mm256_fmadd_pd(_mm256_set1_pd(F.m1[j]),ua1,A);
            y[i][j]=_mm256_mul_pd(_mm256_add_pd(A,S),half);
            if(j<2) y[i][4-j]=_mm256_mul_pd(_mm256_sub_pd(A,S),half);
        }
    }
}

// STUB for optimized kernel
void elast_kd2_psv_opt(
    const double displ_b[][NGLL][NGLL][2][W],
    const double a_b[][NGLL][NGLL][6][W],
    const double displ[][NGLL][NGLL][2],
    const double a[][NGLL][NGLL][6],
    const double H[NGLL][NGLL],
    const double Ht[NGLL][NGLL],
    double f_b[][NGLL][NGLL][2][W],
    double f[][NGLL][NGLL][2],
    int nbatch,
    int tail
) {
    const EOfold Hcol  = fold_cols(H);    // for H * x   (left)
    const EOfold Htcol = fold_cols(Ht);   // for Ht * x  (left)
    const EOfold Hrow  = fold_rows(H);    // for x * H   (right)
    const EOfold Htrow = fold_rows(Ht);   // for x * Ht  (right)

    for(int b=0;b<nbatch;++b){
        __m256d Ux[5][5], Uz[5][5];
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            Ux[i][j]=_mm256_loadu_pd(&displ_b[b][i][j][0][0]);
            Uz[i][j]=_mm256_loadu_pd(&displ_b[b][i][j][1][0]);
        }

        __m256d dUx_dxi[5][5],dUz_dxi[5][5],dUx_deta[5][5],dUz_deta[5][5];
        eo_left (Htcol, Ux, dUx_dxi);    // Ht * Ux
        eo_left (Htcol, Uz, dUz_dxi);    // Ht * Uz
        eo_right(Hrow,  Ux, dUx_deta);   // Ux * H
        eo_right(Hrow,  Uz, dUz_deta);   // Uz * H

        __m256d tmp[5][5], part2[5][5], fx[5][5], fz[5][5];

        // fx = H*(a1*dUx_dxi + a2*dUz_deta)
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            __m256d a1=_mm256_loadu_pd(&a_b[b][i][j][0][0]);
            __m256d a2=_mm256_loadu_pd(&a_b[b][i][j][1][0]);
            tmp[i][j]=_mm256_fmadd_pd(a1,dUx_dxi[i][j],_mm256_mul_pd(a2,dUz_deta[i][j]));
        }
        eo_left(Hcol, tmp, fx);
        // fx += (a4*(dUx_deta+dUz_dxi)) * Ht
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            __m256d a4=_mm256_loadu_pd(&a_b[b][i][j][3][0]);
            tmp[i][j]=_mm256_mul_pd(a4,_mm256_add_pd(dUx_deta[i][j],dUz_dxi[i][j]));
        }
        eo_right(Htrow, tmp, part2);
        for(int i=0;i<5;++i)for(int j=0;j<5;++j)
            fx[i][j]=_mm256_add_pd(fx[i][j],part2[i][j]);

        // fz = H*(a5*dUx_deta + a6*dUz_dxi)
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            __m256d a5=_mm256_loadu_pd(&a_b[b][i][j][4][0]);
            __m256d a6=_mm256_loadu_pd(&a_b[b][i][j][5][0]);
            tmp[i][j]=_mm256_fmadd_pd(a5,dUx_deta[i][j],_mm256_mul_pd(a6,dUz_dxi[i][j]));
        }
        eo_left(Hcol, tmp, fz);
        // fz += (a2*dUx_dxi + a3*dUz_deta) * Ht
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            __m256d a2=_mm256_loadu_pd(&a_b[b][i][j][1][0]);
            __m256d a3=_mm256_loadu_pd(&a_b[b][i][j][2][0]);
            tmp[i][j]=_mm256_fmadd_pd(a2,dUx_dxi[i][j],_mm256_mul_pd(a3,dUz_deta[i][j]));
        }
        eo_right(Htrow, tmp, part2);
        for(int i=0;i<5;++i)for(int j=0;j<5;++j)
            fz[i][j]=_mm256_add_pd(fz[i][j],part2[i][j]);

        for(int i=0;i<5;++i)for(int j=0;j<5;++j){
            _mm256_storeu_pd(&f_b[b][i][j][0][0], fx[i][j]);
            _mm256_storeu_pd(&f_b[b][i][j][1][0], fz[i][j]);
        }
    }
    
    // Tail processing for remainder elements
    int offset = nbatch * W;
    for (int t = 0; t < tail; ++t) {
        elast_kd2_psv_ref_single(displ[offset + t], a[offset + t], H, Ht, f[offset + t]);
    }
}

int NSTEPS = 3208; // default, can be overridden

int main(int argc, char** argv) {
    if (argc > 1) {
        NSTEPS = std::atoi(argv[1]);
        if (NSTEPS <= 0) NSTEPS = 3208;
    }
    double H[NGLL][NGLL];
    double Ht[NGLL][NGLL];
    std::ifstream infile("H_ngll5.txt");
    if (!infile) {
        std::cerr << "Error: H_ngll5.txt not found!\n";
        return 1;
    }
    
    std::string label;
    infile >> label; 
    for (int i = 0; i < NGLL; ++i)
        for (int j = 0; j < NGLL; ++j)
            infile >> H[i][j];
            
    infile >> label; 
    for (int i = 0; i < NGLL; ++i)
        for (int j = 0; j < NGLL; ++j)
            infile >> Ht[i][j];
            
    int nbatch = NELEM / W;
    int tail = NELEM % W;

    auto displ   = new double[NELEM][NGLL][NGLL][2];
    auto a       = new double[NELEM][NGLL][NGLL][6];
    auto f_ref   = new double[NELEM][NGLL][NGLL][2];
    auto f_opt   = new double[NELEM][NGLL][NGLL][2];

    auto displ_b = new double[nbatch][NGLL][NGLL][2][W]();
    auto a_b     = new double[nbatch][NGLL][NGLL][6][W]();
    auto f_opt_b = new double[nbatch][NGLL][NGLL][2][W]();

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int e = 0; e < NELEM; ++e) {
        for (int i = 0; i < NGLL; ++i) {
            for (int j = 0; j < NGLL; ++j) {
                for (int d = 0; d < 2; ++d) displ[e][i][j][d] = dist(rng);
                for (int c = 0; c < 6; ++c) a[e][i][j][c] = dist(rng);
                for (int d = 0; d < 2; ++d) {
                    f_ref[e][i][j][d] = 0.0;
                    f_opt[e][i][j][d] = 0.0;
                }
            }
        }
    }

    // Repack data ONCE
    for (int e = 0; e < NELEM; ++e) {
        if (e < nbatch * W) {
            int b = e / W;
            int lane = e % W;
            for (int i = 0; i < NGLL; ++i) {
                for (int j = 0; j < NGLL; ++j) {
                    for (int d = 0; d < 2; ++d) displ_b[b][i][j][d][lane] = displ[e][i][j][d];
                    for (int c = 0; c < 6; ++c) a_b[b][i][j][c][lane] = a[e][i][j][c];
                }
            }
        }
    }

    std::cout << "Running reference kernel...\n";
    std::vector<double> ref_times;
    for (int run = 0; run < 3; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < NSTEPS; ++step) {
            elast_kd2_psv_ref(displ, a, H, Ht, f_ref);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        ref_times.push_back(diff.count());
    }
    std::sort(ref_times.begin(), ref_times.end());
    double ref_median = ref_times[1];

    std::cout << "Running optimized kernel...\n";
    std::vector<double> opt_times;
    for (int run = 0; run < 3; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < NSTEPS; ++step) {
            elast_kd2_psv_opt(displ_b, a_b, displ, a, H, Ht, f_opt_b, f_opt, nbatch, tail);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        opt_times.push_back(diff.count());
    }
    std::sort(opt_times.begin(), opt_times.end());
    double opt_median = opt_times[1];

    // Unpack f_opt_b into f_opt for correctness check
    for (int b = 0; b < nbatch; ++b) {
        for (int l = 0; l < W; ++l) {
            int e = b * W + l;
            for (int i = 0; i < NGLL; ++i) {
                for (int j = 0; j < NGLL; ++j) {
                    for (int d = 0; d < 2; ++d) {
                        f_opt[e][i][j][d] = f_opt_b[b][i][j][d][l];
                    }
                }
            }
        }
    }

    std::cout << "Sample f values for element 0:\n";
    std::cout << "Node(i,j) Comp | Reference f          | Optimized f        \n";
    std::cout << "---------------------------------------------------------\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int d = 0; d < 2; ++d) {
                std::cout << "(" << i << "," << j << ")     " << d << "    | " 
                          << std::setw(20) << f_ref[0][i][j][d] << " | " 
                          << std::setw(20) << f_opt[0][i][j][d] << "\n";
            }
        }
    }
    std::cout << "\n";

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    
    for (int e = 0; e < NELEM; ++e) {
        for (int i = 0; i < NGLL; ++i) {
            for (int j = 0; j < NGLL; ++j) {
                for (int d = 0; d < 2; ++d) {
                    double val_ref = f_ref[e][i][j][d];
                    double val_opt = f_opt[e][i][j][d];
                    double abs_diff = std::abs(val_ref - val_opt);
                    double rel_diff = abs_diff / (std::abs(val_ref) + 1e-15);
                    if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
                    if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
                }
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "----------------------------------------\n";
    std::cout << "Ref Median Time : " << ref_median << " s\n";
    std::cout << "Opt Median Time : " << opt_median << " s\n";
    std::cout << "Speedup         : " << ref_median / opt_median << "x\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::scientific;
    std::cout << "Max Abs Diff    : " << max_abs_diff << "\n";
    std::cout << "Max Rel Diff    : " << max_rel_diff << "\n";

    delete[] displ;
    delete[] a;
    delete[] f_ref;
    delete[] f_opt;
    delete[] displ_b;
    delete[] a_b;
    delete[] f_opt_b;
    
    return 0;
}
