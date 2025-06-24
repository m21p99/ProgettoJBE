#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// Funzione RLE Encoding ottimizzata
DLL_EXPORT char* rle_encode(const char* input) {
    if (!input || !*input) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    size_t len = strlen(input);
    size_t output_capacity = len * 4 + 1;
    char* output = (char*)malloc(output_capacity);
    if (!output) return NULL;

    size_t out_pos = 0;
    size_t i = 0;

    while (i < len) {
        unsigned char c = (unsigned char)input[i];
        i++;
        
        // Always encode single characters with their count
        // Format: <char><count>#
        int written = snprintf(output + out_pos, output_capacity - out_pos, "%c1#", c);
        if (written < 0 || (size_t)written >= output_capacity - out_pos) {
            output_capacity *= 2;
            char* new_output = (char*)realloc(output, output_capacity);
            if (!new_output) {
                free(output);
                return NULL;
            }
            output = new_output;
            written = snprintf(output + out_pos, output_capacity - out_pos, "%c1#", c);
        }
        out_pos += written;
    }

    output[out_pos] = '\0';
    return output;
}

DLL_EXPORT char* rle_decode(const char* encoded) {
    if (!encoded || !*encoded) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    // First pass: calculate total length
    size_t total_len = 0;
    const char* p = encoded;
    while (*p) {
        char c = *p++;
        size_t count = 0;
        
        // Read count until '#' delimiter
        while (*p && *p != '#') {
            if (!isdigit((unsigned char)*p)) {
                return NULL; // Invalid format
            }
            count = count * 10 + (*p - '0');
            p++;
        }
        
        if (*p != '#') return NULL; // Missing delimiter
        p++; // Skip '#'
        
        total_len += count;
    }

    // Allocate result buffer
    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;

    // Second pass: decode
    char* out = result;
    p = encoded;
    while (*p) {
        char c = *p++;
        size_t count = 0;
        
        while (*p && *p != '#') {
            count = count * 10 + (*p - '0');
            p++;
        }
        p++; // Skip '#'
        
        // Write character 'count' times
        memset(out, c, count);
        out += count;
    }

    *out = '\0';
    return result;
}

// Funzione per liberare la memoria
DLL_EXPORT void free_result(char* ptr) {
    if (ptr) free(ptr);
}

// Test
int main() {
    const char* input = "ciao";  // Stringa con caratteri ripetuti
    char* encoded = rle_encode(input);
    if (encoded) {
        printf("Encoded: %s\n", encoded);
    } else {
        printf("Errore durante l'encoding.\n");
    }

    char* decoded = rle_decode(encoded);
    if (decoded) {
        printf("Decoded: %s\n", decoded);
    } else {
        printf("Errore durante il decoding.\n");
    }
    free_result(encoded);
    free_result(decoded);
    return 0;
}