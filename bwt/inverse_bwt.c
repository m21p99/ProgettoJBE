// inverse_bwt.c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

/*
 * Funzione: inverse_bwt
 * ----------------------
 *  Inverte la trasformazione Burrows-Wheeler su un blocco.
 *
 *  bwt_str:      La stringa risultante dalla BWT (terminata da '\0').
 *  n:            La lunghezza del blocco (numero di caratteri).
 *
 *  Restituisce:  La stringa originale (allocata dinamicamente, da liberare con free()).
 *
 *  Nota: Questa funzione assume che la stringa BWT contenga il carattere terminatore '$'
 *        e calcola automaticamente il suo indice.
 */
DLL_EXPORT char* inverse_bwt(const char* bwt_str, size_t n) {
    if (!bwt_str || n == 0) return NULL;

    // Trova il primo '$' nella stringa: lo usiamo come indice iniziale
    size_t orig_index = 0;
    int found = 0;
    for (size_t i = 0; i < n; i++) {
        if (bwt_str[i] == '$') {
            orig_index = i;
            found = 1;
            break;
        }
    }
    if (!found) {
        // Se non troviamo '$', non possiamo invertire la BWT
        return NULL;
    }

    // Allocazione degli array di supporto
    unsigned char* first_col = (unsigned char*)malloc(n * sizeof(unsigned char));
    if (!first_col) return NULL;
    
    size_t* count = (size_t*)calloc(256, sizeof(size_t));
    size_t* start_pos = (size_t*)calloc(256, sizeof(size_t));
    size_t* LF = (size_t*)malloc(n * sizeof(size_t));
    if (!count || !start_pos || !LF) {
        free(first_col); free(count); free(start_pos); free(LF);
        return NULL;
    }

    // Array temporaneo per le occorrenze
    size_t occurrence[256] = {0};

    // Conta le occorrenze dei caratteri
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) bwt_str[i];
        count[c]++;
    }

    // Calcola le posizioni iniziali (cumulativa)
    size_t sum = 0;
    for (int i = 0; i < 256; i++) {
        start_pos[i] = sum;
        sum += count[i];
        count[i] = 0;  // reimposta count per il prossimo ciclo
    }

    // Costruisci il mapping LF e la "first column"
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) bwt_str[i];
        size_t pos = start_pos[c] + count[c];
        first_col[pos] = c;
        LF[i] = pos;
        count[c]++;
    }

    // Alloca il buffer per la ricostruzione
    char* temp = (char*)malloc((n + 1) * sizeof(char));
    if (!temp) {
        free(first_col); free(count); free(start_pos); free(LF);
        return NULL;
    }
    temp[n] = '\0';

    // Ricostruisci il testo originale seguendo il mapping LF
    size_t pos = orig_index;
    for (size_t i = n; i > 0; i--) {
        temp[i - 1] = bwt_str[pos];
        pos = LF[pos];
    }

    // Copia il risultato in una stringa senza il terminatore '$'
    // Assumiamo che il terminatore appaia una sola volta
    size_t result_len = 0;
    for (size_t i = 0; i < n; i++) {
        if (temp[i] != '$') result_len++;
    }
    char* result = (char*)malloc((result_len + 1) * sizeof(char));
    if (!result) {
        free(first_col); free(count); free(start_pos); free(LF); free(temp);
        return NULL;
    }
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (temp[i] != '$') {
            result[j++] = temp[i];
        }
    }
    result[j] = '\0';

    // Cleanup
    free(first_col);
    free(count);
    free(start_pos);
    free(LF);
    free(temp);

    return result;
}

 // Esempio di funzione main per test (non incluso nella DLL)
 int main() {
     const char* test = "annb$aa";
     size_t n = strlen(test);
     char* orig = inverse_bwt(test, n);
     if (orig) {
         printf("Input: %s\nOutput: %s\n", test, orig);
         free(orig);
     }
     return 0;
 }
