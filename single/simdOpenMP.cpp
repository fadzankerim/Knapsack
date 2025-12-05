#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>     // memcpy, memset
#include <immintrin.h> // AVX2 intrinsics
#include <omp.h>       // OpenMP
#include <cmath>       // std::max
#include <iomanip>     // std::fixed, std::setprecision


// g++ -o knapsackOmp -O3 -mavx2 -fopenmp simdOpenMP.cpp

void generateTestData(int n,
                      std::vector<int> &weights,
                      std::vector<int> &values,
                      int &capacity)
{
    weights.resize(n);
    values.resize(n);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < n; ++i)
    {
        weights[i] = std::rand() % 100 + 1;
        values[i] = std::rand() % 500 + 50;
    }
    capacity = n * 50; 
}

class ContainerLoader
{
private:
    std::vector<int> weights;
    std::vector<int> values;
    int capacity;
    int n;

public:
    ContainerLoader(const std::vector<int> &w,
                    const std::vector<int> &v,
                    int cap)
        : weights(w), values(v), capacity(cap), n((int)w.size()) {}

    
    int knapsackSIMD_stream_omp(long long &simd_omp_duration_us, int numThreads) const
    {
        if (capacity <= 0) return 0;

        const int N = capacity + 8;
        int *prev = new int[N];
        int *dp = new int[N];

        std::memset(prev, 0, sizeof(int) * N);
        std::memset(dp, 0, sizeof(int) * N);

        simd_omp_duration_us = 0; 

        for (int i = 0; i < n; ++i)
        {
            const int wi = weights[i];
            const int vi = values[i];

            int *tmp = prev;
            prev = dp;
            dp = tmp;

            if (wi > 0)
            {
                const int count = std::min(wi, capacity + 1);
                std::memcpy(dp, prev, sizeof(int) * count);
            }

            __m256i add_vi = _mm256_set1_epi32(vi);

            const int Wlim = capacity - 7;
            const int w_start = wi;
            
            double omp_start_time = omp_get_wtime();
            
            // koristi onoliko threadova koliko je proslijedjenjo funkcij
            #pragma omp parallel num_threads(numThreads) 
            {
                int *__restrict loc_prev = prev;
                int *__restrict loc_dp = dp;
                __m256i loc_add_vi = add_vi;

                #pragma omp for schedule(static)
                for (int w = w_start; w <= Wlim; w += 8)
                {
                    __m256i prev_curr =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&loc_prev[w]));
                    __m256i prev_shift =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&loc_prev[w - wi]));

                    __m256i cand = _mm256_add_epi32(prev_shift, loc_add_vi);
                    __m256i res = _mm256_max_epi32(prev_curr, cand);

                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&loc_dp[w]), res);
                }
            } 
            
            double omp_end_time = omp_get_wtime();
            simd_omp_duration_us += (long long)((omp_end_time - omp_start_time) * 1000000.0);

            // tail skalarno
            int w_tail = std::max(w_start, Wlim + 1);
            for (int w = w_tail; w <= capacity; ++w)
            {
                int take = prev[w - wi] + vi;
                int skip = prev[w];
                dp[w] = std::max(take, skip);
            }
        }

        int result = dp[capacity];

        delete[] prev;
        delete[] dp;

        return result;
    }
};

int main(){
    
    const int N_ITEMS = 8000; 
    
    std::vector<int> weights;
    std::vector<int> values;
    int capacity;
    
    generateTestData(N_ITEMS, weights, values, capacity);
    
    ContainerLoader loader(weights, values, capacity);

    // Ovdje fiksiramo broj threadova
    const int FIXED_THREADS = 2;

    std::cout << "Broj predmeta (n): " << N_ITEMS << "\n";
    std::cout << "Generirani kapacitet (W): " << capacity << "\n";
    std::cout << "Max hardware threads dostupno: " << omp_get_max_threads() << "\n";
    std::cout << "Podeseno da koristi fiksnih: " << FIXED_THREADS << " threadova.\n";
    std::cout << "----------------------------------------------------------------\n";

    long long simd_omp_time_us = 0; 
    
    auto start_time_total = std::chrono::high_resolution_clock::now();
    
    // pozia se funkcija sa brojem thredova
    int max_value = loader.knapsackSIMD_stream_omp(simd_omp_time_us, FIXED_THREADS);
    
    auto end_time_total = std::chrono::high_resolution_clock::now();
    auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time_total - start_time_total).count();

    double total_ms = (double)total_duration_us / 1000.0;
    double simd_omp_ms = (double)simd_omp_time_us / 1000.0;
    
    double percentage = 0.0;
    if (total_duration_us > 0) {
        percentage = (double)simd_omp_time_us * 100.0 / total_duration_us;
    }
    
    std::cout << "Maksimalna ukupna vrijednost: " << max_value << "\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Ukupno vrijeme: " 
              << std::fixed << std::setprecision(3) 
              << total_ms << " ms\n";
    std::cout << "Vrijeme OpenMP + SIMD dijela: " 
              << simd_omp_ms << " ms\n";
    std::cout << "Udio paralelnog dijela: "
              << std::fixed << std::setprecision(2)
              << percentage << " %\n";
    std::cout << "----------------------------------------------------------------\n";

    return 0;
}