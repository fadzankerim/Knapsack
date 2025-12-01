import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Postavke za vizualizaciju ---
CSV_FILE = 'results.csv'
BASE_METHOD_NAME = '1D_baseline'

def visualize_results():
    """
    Učitava rezultate iz CSV fajla i kreira grafičku vizualizaciju.
    """
    if not os.path.exists(CSV_FILE):
        print(f"🛑 Greška: Datoteka '{CSV_FILE}' nije pronađena.")
        print("Uvjerite se da ste pokrenuli C++ program s odgovarajućim argumentima:")
        print("  $ g++ -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17 knapsackOptimized.cpp -o knap")
        print("  $ ./knap")
        return

    # Učitavanje podataka
    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        print(f"🛑 Greška: Datoteka '{CSV_FILE}' je prazna. Provjerite C++ izlaz.")
        return
    
    # ----------------------------------------------------
    # 1. Grafikon apsolutnih vremena
    # ----------------------------------------------------
    
    plt.figure(figsize=(10, 6))
    
    # Određivanje boja
    colors = ['#88c0d0', '#5e81ac', '#a3be8c', '#bf616a']
    
    bars = plt.bar(df['Method'], df['TimeSeconds'], color=colors)
    
    plt.title(f'Apsolutno Vrijeme Izvršavanja (Sekunde)', fontsize=16)
    plt.ylabel('Vrijeme (sekunde)', fontsize=12)
    plt.xlabel('Metoda', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dodavanje teksta iznad stupaca
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + yval*0.05, 
                 f'{yval:.4f} s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('knapsack_vrijeme.png')
    print("✅ Generisan grafikon apsolutnog vremena: knapsack_vrijeme.png")
    
    # ----------------------------------------------------
    # 2. Grafikon ubrzanja vs. 1D_baseline
    # ----------------------------------------------------
    
    # Filtriranje 1D_baseline metode, jer je njeno ubrzanje 1.0x
    df_speedup = df[df['Method'] != BASE_METHOD_NAME].copy()
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(df_speedup['Method'], df_speedup['SpeedupVs1DBaseline'], color=colors[::2])
    
    plt.title(f'Ubrzanje u Odnosu na {BASE_METHOD_NAME} (Referentna)', fontsize=16)
    plt.ylabel('Faktor Ubrzanja (X)', fontsize=12)
    plt.xlabel('Metoda', fontsize=12)
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label=f'{BASE_METHOD_NAME} (1.0x)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dodavanje teksta iznad stupaca
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, 
                 f'{yval:.2f}x', ha='center', va='bottom', fontsize=10, 
                 fontweight='bold', color='black')

    plt.legend()
    plt.tight_layout()
    plt.savefig('knapsack_ubrzanje.png')
    print("✅ Generisan grafikon ubrzanja: knapsack_ubrzanje.png")
    
    # Prikazivanje grafikona (opcionalno, ako se pokreće u interaktivnom okruženju)
    # plt.show()


if __name__ == "__main__":
    visualize_results()