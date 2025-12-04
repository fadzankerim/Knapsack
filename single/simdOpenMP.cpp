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

// Funkcija za generiranje testnih podataka
void generateTestData(int n,
                      std::vector<int> &weights,
                      std::vector<int> &values,
                      int &capacity)
{
    weights.resize(n);
    values.resize(n);

    // Inicijalizacija generatora slučajnih brojeva
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < n; ++i)
    {
        weights[i] = std::rand() % 100 + 1; // Težina: 1..100
        values[i] = std::rand() % 500 + 50; // Vrijednost: 50..549
    }
    // Kapacitet: n * 50
    capacity = n * 50; 
}

//  ContainerLoader klasa

class ContainerLoader
{
private:
    std::vector<int> weights; // tezine
    std::vector<int> values;  // vrijednosti
    int capacity;             // kapacitet
    int n;                    // br predmeta

public:
    ContainerLoader(const std::vector<int> &w,
                    const std::vector<int> &v,
                    int cap)
        : weights(w), values(v), capacity(cap), n((int)w.size()) {}


    
    int knapsackSIMD_stream_omp(long long &simd_omp_duration_us) const
    {
        if (capacity <= 0) return 0;

        const int N = capacity + 8;
        int *prev = new int[N];
        int *dp = new int[N];

        std::memset(prev, 0, sizeof(int) * N);
        std::memset(dp, 0, sizeof(int) * N);

        // reset vremena
        simd_omp_duration_us = 0; 

        for (int i = 0; i < n; ++i)
        {
            const int wi = weights[i];
            const int vi = values[i];

            // swap pointera
            int *tmp = prev;
            prev = dp;
            dp = tmp;

            // Kopiranje starog stanja za kapacitete < wi
            if (wi > 0)
            {
                const int count = std::min(wi, capacity + 1);
                std::memcpy(dp, prev, sizeof(int) * count);
            }

            __m256i add_vi = _mm256_set1_epi32(vi);

            const int Wlim = capacity - 7;
            const int w_start = wi;


            
            // mjerenje vremena paralelne regije
            double omp_start_time = omp_get_wtime();
            
            #pragma omp parallel
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
            } // Kraj omp parallel regije
            
            double omp_end_time = omp_get_wtime();
            
            
            simd_omp_duration_us += (long long)((omp_end_time - omp_start_time) * 1000000.0);
            


            // Skalarni Tail 
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

}; // Kraj klase ContainerLoader


int main(){
    
    const int N_ITEMS = 8000; 
    
    std::vector<int> weights;
    std::vector<int> values;
    int capacity;
    
    
    generateTestData(N_ITEMS, weights, values, capacity);
    
    ContainerLoader loader(weights, values, capacity);

    std::cout << "Broj predmeta (n): " << N_ITEMS << "\n";
    std::cout << "Generirani kapacitet (W): " << capacity << "\n";
    std::cout << "Broj dostupnih niti (max threads): " << omp_get_max_threads() << "\n";
    std::cout << "----------------------------------------------------------------\n";

    
    
    long long simd_omp_time_us = 0; // Vrijeme provedeno u paraleliziranom SIMD bloku
    
    // Mjerenje UKUPNOG VREMENA FUNKCIJE
    auto start_time_total = std::chrono::high_resolution_clock::now();
    
    // Izvršavanje funkcije i vraćanje kumulativnog SIMD/OpenMP vremena
    int max_value = loader.knapsackSIMD_stream_omp(simd_omp_time_us);
    
    auto end_time_total = std::chrono::high_resolution_clock::now();
    // Ukupno vrijeme izvršavanja funkcije u mikrosekundama
    auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time_total - start_time_total).count();

    
    
    double total_ms = (double)total_duration_us / 1000.0;
    double simd_omp_ms = (double)simd_omp_time_us / 1000.0;
    
    // Izračun postotka
    double percentage = 0.0;
    if (total_duration_us > 0) {
        percentage = (double)simd_omp_time_us * 100.0 / total_duration_us;
    }
    
    //rez
    std::cout << "Maksimalna ukupna vrijednost: " << max_value << "\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "ukupno vrijeme izvršavanja funkcije: " 
              << std::fixed << std::setprecision(3) 
              << total_ms << " milisekundi\n";
    std::cout << "Vrijeme izvršavanja OpenMP + SIMD bloka (kumulativno): " 
              << simd_omp_ms << " milisekundi\n";
    std::cout << "\nPROCENTUALNO VRIJEME KRITIČNE SEKCIJE: "
              << std::fixed << std::setprecision(2)
              << percentage << " %\n";
    std::cout << "----------------------------------------------------------------\n";

    return 0;
}