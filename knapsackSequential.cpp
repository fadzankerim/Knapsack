// knapsack_simd.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>   // AVX2 intrinsics
#include <limits>

class ContainerLoader {
private:
    std::vector<int> weights;  // Težine/volumeni dobara
    std::vector<int> values;   // Vrijednosti dobara
    int capacity;              // Kapacitet kontejnera
    int n;                     // Broj dobara

public:
    ContainerLoader(const std::vector<int>& w, const std::vector<int>& v, int cap)
        : weights(w), values(v), capacity(cap), n((int)w.size()) {}

    // 2D DP (klasično)
    int knapsackClassic() {
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = std::max(dp[i - 1][w],
                                        dp[i - 1][w - weights[i - 1]] + values[i - 1]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }

    // 1D DP (sekvencijalno, ispravno)
    int knapsackOptimized() {
        std::vector<int> dp(capacity + 1, 0);
        for (int i = 0; i < n; i++) {
            for (int w = capacity; w >= weights[i]; w--) {
                dp[w] = std::max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
        return dp[capacity];
    }

    
    // SIMD (AVX2) – koristi dvostruki bafer (prev -> dp), gather + maska (radi poređenja)
    int knapsackSIMD() {
        const int aligned = ((capacity + 7) / 8) * 8;
        
        std::vector<int> dp(aligned + 8, 0);
        std::vector<int> prev(aligned + 8, 0);
        
        for (int i = 0; i < n; i++) {
            const int wi = weights[i];
            const int vi = values[i];
            
            std::swap(dp, prev);
            
            // obrađujemo u blokovima po 8 "w" vrijednosti: w..w+7
            for (int w = 0; w <= aligned; w += 8) {
                const int upto = std::min(w + 7, capacity);
                if (w > upto) continue;
                
                // 1) učitaj prev[w..w+7]
                __m256i prev_vec = _mm256_loadu_si256((const __m256i*)&prev[w]);
                
                // 2) indeksi (w..w+7) - wi
                int idxs_arr[8];
                for (int j = 0; j < 8; ++j) idxs_arr[j] = (w + j) - wi;
                __m256i idxs = _mm256_loadu_si256((const __m256i*)idxs_arr);
                
                // validnost: idx >= 0 && (w+j) <= capacity
                __m256i ge0 = _mm256_cmpgt_epi32(idxs, _mm256_set1_epi32(-1));
                __m256i wj   = _mm256_add_epi32(_mm256_set1_epi32(w),
                _mm256_setr_epi32(0,1,2,3,4,5,6,7));
                __m256i leC  = _mm256_cmpgt_epi32(_mm256_set1_epi32(capacity+1), wj);
                __m256i valid = _mm256_and_si256(ge0, leC);
                
                // 3) gather: prev[idxs]  (skala 4 bajta jer su int)
                __m256i gathered = _mm256_i32gather_epi32(&prev[0], idxs, 4);
                
                // kandidat = gathered + vi
                __m256i cand = _mm256_add_epi32(gathered, _mm256_set1_epi32(vi));

                // invalidne pozicije -> -INF (da ih max ignoriše)
                __m256i negInf = _mm256_set1_epi32(std::numeric_limits<int>::min() / 2);
                cand = _mm256_blendv_epi8(negInf, cand, valid);
                
                // 4) max(prev[w], cand)
                __m256i res = _mm256_max_epi32(prev_vec, cand);
                
                // 5) upis (čuvaj kraj preko capacity skalarno)
                if (w + 7 <= capacity) {
                    _mm256_storeu_si256((__m256i*)&dp[w], res);
                } else {
                    alignas(32) int tmp[8];
                    _mm256_store_si256((__m256i*)tmp, res);
                    for (int j = 0; w + j <= capacity && j < 8; ++j) dp[w + j] = tmp[j];
                }
            }
        }
        return dp[capacity];
    }

    // 1D DP SIMD (AVX2, bez gather-a) — dva bafera, naprijed po w


    int knapsackOptimizedSIMD() {
        std::vector<int> dp(capacity + 8, 0), prev(capacity + 8, 0);
    
        for (int i = 0; i < n; ++i) {
            const int wi = weights[i];
            const int vi = values[i];
            std::swap(dp, prev);
    
            // ispod wi: predmet ne može stati -> kopiraj staro stanje
            if (wi > 0) std::copy(prev.begin(), prev.begin() + wi, dp.begin());
    
            __m256i add_vi = _mm256_set1_epi32(vi);
    
            int w = wi;
            for (; w + 7 <= capacity; w += 8) {
                // sekvencijalna čitanja (brzo): prev[w..w+7] i prev[w-wi..w-wi+7]
                __m256i prev_curr  = _mm256_loadu_si256((const __m256i*)&prev[w]);
                __m256i prev_shift = _mm256_loadu_si256((const __m256i*)&prev[w - wi]);
    
                __m256i cand = _mm256_add_epi32(prev_shift, add_vi);
                __m256i res  = _mm256_max_epi32(prev_curr, cand);
    
                _mm256_storeu_si256((__m256i*)&dp[w], res);
            }
            // rep (tail) skalarno
            for (; w <= capacity; ++w) {
                int take = prev[w - wi] + vi;
                int skip = prev[w];
                dp[w] = (take > skip) ? take : skip;
            }
        }
        return dp[capacity];
    }
};

// Generisanje test podataka
void generateTestData(int n, std::vector<int>& weights, std::vector<int>& values, int& capacity) {
    weights.resize(n);
    values.resize(n);
    
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < n; i++) {
        weights[i] = std::rand() % 100 + 1;
        values[i]  = std::rand() % 500 + 50;
    }
    capacity = n * 50;
}

int main() {
    const int num_items = 1000;

    std::vector<int> weights, values;
    int capacity;
    generateTestData(num_items, weights, values, capacity);

    ContainerLoader loader(weights, values, capacity);

    std::cout << "SEKVENCIJALNA VERZIJA - POREĐENJE KNAPSACK METODA\n";
    std::cout << "Broj predmeta: " << num_items << ", Kapacitet: " << capacity << "\n\n";

    // Mjerenje vremena
    clock_t start, end;

    start = clock();
    int result_classic = loader.knapsackClassic();
    end = clock();
    double time_classic = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    int result_opt = loader.knapsackOptimized();
    end = clock();
    double time_opt = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    int result_simd_gather = loader.knapsackSIMD();
    end = clock();
    double time_simd_gather = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    int result_opt_simd = loader.knapsackOptimizedSIMD();
    end = clock();
    double time_opt_simd = double(end - start) / CLOCKS_PER_SEC;

    // Rezultati
    std::cout << "Rezultati:\n";
    std::cout << "Klasični DP:               " << result_classic    << " (Vrijeme: " << time_classic     << " s)\n";
    std::cout << "Optimizovani 1D:           " << result_opt        << " (Vrijeme: " << time_opt         << " s)\n";
    std::cout << "SIMD (gather + maska):     " << result_simd_gather<< " (Vrijeme: " << time_simd_gather << " s)\n";
    std::cout << "Optimizovani 1D SIMD:      " << result_opt_simd   << " (Vrijeme: " << time_opt_simd    << " s)\n";

    // Provjera konzistentnosti
    if (result_classic == result_opt &&
        result_opt     == result_simd_gather &&
        result_simd_gather == result_opt_simd) {
        std::cout << "\n✓ Rezultati su konzistentni!\n";
    } else {
        std::cout << "\n⚠ Upozorenje: Rezultati se razlikuju!\n";
    }

    // Uporedna ubrzanja vs klasični 2D
    std::cout << "\nUbrzanje (vs Classic):\n";
    std::cout << "Optimizovani 1D:           " << time_classic / time_opt        << "x\n";
    std::cout << "SIMD (gather + mask):      " << time_classic / time_simd_gather<< "x\n";
    std::cout << "Optimizovani 1D SIMD (NG): " << time_classic / time_opt_simd   << "x\n";

    return 0;
}
