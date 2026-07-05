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
const int NSTEPS = 3208;
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
                            + ( a[i][k][3]*dUx_deta[i][k] + a[i][k][4]*dUz_dxi[i][k] ) * Ht[k][j];
                               
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
    for (int b = 0; b < nbatch; ++b) {
        // gradients: [i][j] each a __m256d (4 elements per lane)
        __m256d dUx_dxi[5][5], dUz_dxi[5][5], dUx_deta[5][5], dUz_deta[5][5];

        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d gx_xi  = _mm256_setzero_pd();
                __m256d gz_xi  = _mm256_setzero_pd();
                __m256d gx_eta = _mm256_setzero_pd();
                __m256d gz_eta = _mm256_setzero_pd();
                for (int k = 0; k < 5; ++k) {
                    // left-multiply  Ht*U : scalar Ht[i][k], vector U[k][j]
                    __m256d ht = _mm256_set1_pd(Ht[i][k]);
                    gx_xi = _mm256_fmadd_pd(ht, _mm256_loadu_pd(&displ_b[b][k][j][0][0]), gx_xi);
                    gz_xi = _mm256_fmadd_pd(ht, _mm256_loadu_pd(&displ_b[b][k][j][1][0]), gz_xi);
                    // right-multiply U*H : scalar H[k][j], vector U[i][k]
                    __m256d hh = _mm256_set1_pd(H[k][j]);
                    gx_eta = _mm256_fmadd_pd(hh, _mm256_loadu_pd(&displ_b[b][i][k][0][0]), gx_eta);
                    gz_eta = _mm256_fmadd_pd(hh, _mm256_loadu_pd(&displ_b[b][i][k][1][0]), gz_eta);
                }
                dUx_dxi[i][j]  = gx_xi;
                dUz_dxi[i][j]  = gz_xi;
                dUx_deta[i][j] = gx_eta;
                dUz_deta[i][j] = gz_eta;
            }

        // forces (nelast=6). a index map: a1..a6 -> a_b[...][0..5][...]
        __m256d tmp[5][5];

        // f_x part 1: tmp = a1*dUx_dxi + a2*dUz_deta ; fx = H*tmp
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d a1 = _mm256_loadu_pd(&a_b[b][i][j][0][0]);
                __m256d a2 = _mm256_loadu_pd(&a_b[b][i][j][1][0]);
                tmp[i][j] = _mm256_fmadd_pd(a1, dUx_dxi[i][j],
                            _mm256_mul_pd(a2, dUz_deta[i][j]));
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d acc = _mm256_setzero_pd();
                for (int k = 0; k < 5; ++k)  // H*tmp : scalar H[i][k], vector tmp[k][j]
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(H[i][k]), tmp[k][j], acc);
                _mm256_storeu_pd(&f_b[b][i][j][0][0], acc);
            }
        // f_x part 2: tmp = a4*(dUx_deta + dUz_dxi) ; fx += tmp*Ht
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d a4 = _mm256_loadu_pd(&a_b[b][i][j][3][0]);
                tmp[i][j] = _mm256_mul_pd(a4, _mm256_add_pd(dUx_deta[i][j], dUz_dxi[i][j]));
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d acc = _mm256_loadu_pd(&f_b[b][i][j][0][0]);
                for (int k = 0; k < 5; ++k)  // tmp*Ht : scalar Ht[k][j], vector tmp[i][k]
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(Ht[k][j]), tmp[i][k], acc);
                _mm256_storeu_pd(&f_b[b][i][j][0][0], acc);
            }

        // f_z part 1: tmp = a5*dUx_deta + a6*dUz_dxi ; fz = H*tmp
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d a5 = _mm256_loadu_pd(&a_b[b][i][j][4][0]);
                __m256d a6 = _mm256_loadu_pd(&a_b[b][i][j][5][0]);
                tmp[i][j] = _mm256_fmadd_pd(a5, dUx_deta[i][j],
                            _mm256_mul_pd(a6, dUz_dxi[i][j]));
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d acc = _mm256_setzero_pd();
                for (int k = 0; k < 5; ++k)  // H*tmp
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(H[i][k]), tmp[k][j], acc);
                _mm256_storeu_pd(&f_b[b][i][j][1][0], acc);
            }
        // f_z part 2: tmp = a2*dUx_dxi + a3*dUz_deta ; fz += tmp*Ht
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d a2 = _mm256_loadu_pd(&a_b[b][i][j][1][0]);
                __m256d a3 = _mm256_loadu_pd(&a_b[b][i][j][2][0]);
                tmp[i][j] = _mm256_fmadd_pd(a2, dUx_dxi[i][j],
                            _mm256_mul_pd(a3, dUz_deta[i][j]));
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                __m256d acc = _mm256_loadu_pd(&f_b[b][i][j][1][0]);
                for (int k = 0; k < 5; ++k)  // tmp*Ht
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(Ht[k][j]), tmp[i][k], acc);
                _mm256_storeu_pd(&f_b[b][i][j][1][0], acc);
            }
    }
    
    // Tail processing for remainder elements
    int offset = nbatch * W;
    for (int t = 0; t < tail; ++t) {
        elast_kd2_psv_ref_single(displ[offset + t], a[offset + t], H, Ht, f[offset + t]);
    }
}

int main() {
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
