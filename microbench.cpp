#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

const int NGLL = 5;
const int NELEM = 12800;
const int NSTEPS = 3208;

// REFERENCE kernel: scalar translation of ELAST_KD2_PSV (nelast=6)
void elast_kd2_psv_ref(
    const double displ[NELEM][NGLL][NGLL][2],
    const double a[NELEM][NGLL][NGLL][6],
    const double H[NGLL][NGLL],
    const double Ht[NGLL][NGLL],
    double f[NELEM][NGLL][NGLL][2]
) {
    for (int e = 0; e < NELEM; ++e) {
        double dUx_dxi[NGLL][NGLL] = {{0}};
        double dUz_dxi[NGLL][NGLL] = {{0}};
        double dUx_deta[NGLL][NGLL] = {{0}};
        double dUz_deta[NGLL][NGLL] = {{0}};

        for (int j = 0; j < NGLL; ++j) {
            for (int k = 0; k < NGLL; ++k) {
                for (int i = 0; i < NGLL; ++i) {
                    dUx_dxi[i][j]  += Ht[i][k] * displ[e][k][j][0];
                    dUz_dxi[i][j]  += Ht[i][k] * displ[e][k][j][1];
                    dUx_deta[i][j] += displ[e][i][k][0] * H[k][j];
                    dUz_deta[i][j] += displ[e][i][k][1] * H[k][j];
                }
            }
        }

        for (int i = 0; i < NGLL; ++i) {
            for (int j = 0; j < NGLL; ++j) {
                f[e][i][j][0] = 0.0;
                f[e][i][j][1] = 0.0;
            }
        }

        for (int j = 0; j < NGLL; ++j) {
            for (int k = 0; k < NGLL; ++k) {
                for (int i = 0; i < NGLL; ++i) {
                    f[e][i][j][0] += H[i][k] * ( a[e][k][j][0]*dUx_dxi[k][j] + a[e][k][j][1]*dUz_deta[k][j] )
                                   + ( a[e][i][k][3]*dUx_deta[i][k] + a[e][i][k][4]*dUz_dxi[i][k] ) * Ht[k][j];
                                   
                    f[e][i][j][1] += H[i][k] * ( a[e][k][j][4]*dUx_deta[k][j] + a[e][k][j][5]*dUz_dxi[k][j] )
                                   + ( a[e][i][k][1]*dUx_dxi[i][k] + a[e][i][k][2]*dUz_deta[i][k] ) * Ht[k][j];
                }
            }
        }
    }
}

// STUB for optimized kernel
void elast_kd2_psv_opt(
    const double displ[NELEM][NGLL][NGLL][2],
    const double a[NELEM][NGLL][NGLL][6],
    const double H[NGLL][NGLL],
    const double Ht[NGLL][NGLL],
    double f[NELEM][NGLL][NGLL][2]
) {
    // Mohit: even-odd + W=4 element-batched AVX2 kernel goes here
    // Initially just calling the reference so the harness runs
    elast_kd2_psv_ref(displ, a, H, Ht, f);
}

int main() {
    // Read H and Ht from H_ngll5.txt
    double H[NGLL][NGLL];
    double Ht[NGLL][NGLL];
    std::ifstream infile("H_ngll5.txt");
    if (!infile) {
        std::cerr << "Error: H_ngll5.txt not found!\n";
        return 1;
    }
    
    std::string label;
    infile >> label; // "H"
    for (int i = 0; i < NGLL; ++i)
        for (int j = 0; j < NGLL; ++j)
            infile >> H[i][j];
            
    infile >> label; // "Ht"
    for (int i = 0; i < NGLL; ++i)
        for (int j = 0; j < NGLL; ++j)
            infile >> Ht[i][j];
            
    // Allocate arrays on heap to avoid stack overflow
    auto displ = new double[NELEM][NGLL][NGLL][2];
    auto a     = new double[NELEM][NGLL][NGLL][6];
    auto f_ref = new double[NELEM][NGLL][NGLL][2];
    auto f_opt = new double[NELEM][NGLL][NGLL][2];

    // Initialize with bounded pseudo-random doubles
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

    // Timing loop for REF
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

    // Timing loop for OPT
    std::cout << "Running optimized kernel...\n";
    std::vector<double> opt_times;
    for (int run = 0; run < 3; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < NSTEPS; ++step) {
            elast_kd2_psv_opt(displ, a, H, Ht, f_opt);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        opt_times.push_back(diff.count());
    }
    std::sort(opt_times.begin(), opt_times.end());
    double opt_median = opt_times[1];

    // Correctness check
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
    
    return 0;
}
