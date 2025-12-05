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



// g++ -o knapsackSimd -O3 -mavx2 simdKnapsack.cpp


// generisanje testnih podataka
void generateTestData(int n,
                      std::vector<int> &weights,
                      std::vector<int> &values,
                      int &capacity)
{
    weights.resize(n);
    values.resize(n);

    // generator slucajnih brojeva
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < n; ++i)
    {
        weights[i] = std::rand() % 100 + 1; // Težina: 1 do 100
        values[i] = std::rand() % 500 + 50; // Vrijednost: 50 do 549
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


    
    int knapsackSIMD_stream(long long &simd_duration_us) const
    {
        if (capacity <= 0) return 0;
        
        const int N = capacity + 8;

        int *prev = new int[N];
        int *dp = new int[N];

        std::memset(prev, 0, sizeof(int) * N);
        std::memset(dp, 0, sizeof(int) * N);
        
        // Inicijalizacija kumulativnog vremena
        simd_duration_us = 0;

        for (int i = 0; i < n; ++i)
        {
            const int wi = weights[i];
            const int vi = values[i];

            // Swap pointera
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
            int w = wi;
            const int Wlim = capacity - 7; 

            auto simd_start = std::chrono::high_resolution_clock::now(); 
            
            // SIMD Petlja (glavni dio)
            for (; w <= Wlim; w += 8)
            {
                // Učitavanje prev[w..w+7] (skip)
                __m256i prev_curr = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&prev[w]));
                // Učitavanje prev[w-wi..w-wi+7] (take, dio)
                __m256i prev_shift = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&prev[w - wi]));

                // take = prev_shift + vi
                __m256i cand = _mm256_add_epi32(prev_shift, add_vi);
                // max(skip, take)
                __m256i res = _mm256_max_epi32(prev_curr, cand);

                // Pohrana rezultata u dp[w..w+7]
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dp[w]), res);
            }
            
            auto simd_end = std::chrono::high_resolution_clock::now();
            // Dodavanje trajanja SIMD bloka na ukupan zbroj
            simd_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(simd_end - simd_start).count();

            // Skalarni Tail (za preostale indekse)
            for (; w <= capacity; ++w)
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

    std::cout << "Broj predmeta (n): " << N_ITEMS << "\n";
    std::cout << "Generisani kapacitet (W = n * 50): " << capacity << "\n";
    std::cout << "-------------------------------------------------------\n";

    // Mjera vjremena provedenog u simd bloku i ukupno vrijeme izvrsavanja programa
    
    long long simd_time_us = 0; // Vrijeme provedeno u SIMD petlji (mikrosekunde)
    
    // Mjerenje UKUPNOG VREMENA FUNKCIJE
    auto start_time_total = std::chrono::high_resolution_clock::now();
    
    // Izvršavanje funkcije i vraćanje kumulativnog SIMD vremena pomoću reference
    int max_value = loader.knapsackSIMD_stream(simd_time_us);
    
    auto end_time_total = std::chrono::high_resolution_clock::now();
    // Ukupno vrijeme izvršavanja funkcije u mikrosekundama
    auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time_total - start_time_total).count();

    
    double total_ms = (double)total_duration_us / 1000.0;
    double simd_ms = (double)simd_time_us / 1000.0;
    
    // izracun postotka: (SIMD Vrijeme / Ukupno Vrijeme) * 100
    double percentage = 0.0;
    if (total_duration_us > 0) {
        percentage = (double)simd_time_us * 100.0 / total_duration_us;
    }
    
    std::cout << "Maksimalna ukupna vrijednost: " << max_value << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Ukupno vrijeme izvršavanja funkcije: " 
              << std::fixed << std::setprecision(3) 
              << total_ms << " milisekundi\n";
    std::cout << "Vrijeme izvršavanja SIMD petlje (kumulativno): " 
              << simd_ms << " milisekundi\n";
    std::cout << "\nROCENTUALNO VRIJEME SIMD DIJELA:"
              << std::fixed << std::setprecision(2)
              << percentage << " %\n";
    std::cout << "-------------------------------------------------------\n";

    return 0;
}