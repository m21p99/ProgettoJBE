import numpy as np
import struct, os
from collections import Counter, namedtuple
from decimal import getcontext, Decimal
from fractions import Fraction
import multiprocessing

# Impostazione della precisione per eventuali calcoli (non più usata per l'aritmetica)
getcontext().prec = 100

# Utilizziamo EncodedData per ciascun blocco
EncodedData = namedtuple("EncodedData", ["encoded_value", "total_bits", "total", "ranges", "length"])

# ============================
# FUNZIONI DI RUN-LENGTH
# ============================
def run_length_encoding(data):
    """Applica Run-Length Encoding (RLE) a una stringa."""
    encoded = []
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            i += 1
            count += 1
        encoded.append((data[i], count))
        i += 1
    return encoded

def run_length_decoding(encoded):
    """Decodifica una stringa codificata con RLE."""
    decoded = []
    i = 0
    while i < len(encoded):
        char = encoded[i]
        i += 1
        count_str = ""
        while i < len(encoded) and encoded[i].isdigit():
            count_str += encoded[i]
            i += 1
        count = int(count_str) if count_str else 1
        decoded.extend([char] * count)
    return "".join(decoded)

# ============================
# FUNZIONI DI BURROWS-WHEELER (BWT)
# ============================
def parallel_bwt(text, block_size=1024):
    """Applica BWT in parallelo su blocchi."""
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    with multiprocessing.Pool() as pool:
        results = pool.map(burrows_wheeler_transform, blocks)
    return "".join(results)

def burrows_wheeler_transform(text, block_size=1024):
    """Applica la trasformazione di Burrows-Wheeler (BWT)."""
    text += "$"
    if len(text) <= block_size:
        rotations = sorted([text[i:] + text[:i] for i in range(len(text))])
        result = "".join(row[-1] for row in rotations)
        return result

    output = []
    for i in range(0, len(text), block_size):
        block = text[i:i+block_size]
        rotations = sorted([block[j:] + block[:j] for j in range(len(block))])
        transformed_block = "".join(row[-1] for row in rotations)
        output.append(transformed_block)
    return "".join(output)

def inverse_burrows_wheeler(bwt_string):
    """Ricostruisce la stringa originale dalla trasformata di Burrows-Wheeler."""
    table = ["" for _ in bwt_string]
    for _ in range(len(bwt_string)):
        table = sorted([bwt_string[i] + table[i] for i in range(len(bwt_string))])
    for row in table:
        if row.endswith("$"):
            return row[:-1]

# ============================
# FUNZIONI DI MOVE-TO-FRONT (MTF)
# ============================
def parallel_mtf(text, block_size=1024):
    """Esegue Move-To-Front Transform in parallelo su blocchi."""
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    with multiprocessing.Pool() as pool:
        results = pool.map(move_to_front_transform, blocks)
    return [item for sublist in results for item in sublist]

def move_to_front_transform(text):
    """Applica Move-To-Front Transform (MTF) su una stringa."""
    alphabet = sorted(set(text))
    encoded = []
    for char in text:
        index = alphabet.index(char)
        encoded.append(index)
        alphabet.insert(0, alphabet.pop(index))
    return encoded

def move_to_front_decoding(encoded, text):
    """Decodifica la sequenza MTF usando l'alfabeto iniziale."""
    alphabet = sorted(set(text))
    decoded = []
    for index in encoded:
        symbol = alphabet[index]
        decoded.append(symbol)
        alphabet.pop(index)
        alphabet.insert(0, symbol)
    return "".join(decoded)

# ============================
# FUNZIONI DI J-BIT ENCODING/DECODING
# ============================
def jbit_encoding(data):
    """Applica J-bit Encoding separando i dati e la mappa dei bit."""
    data_I = []  # Valori non zero
    data_II = []  # Mappa dei bit
    temp_byte = 0
    bit_count = 0
    count = 0
    for value in data:
        if value != 0:
            data_I.append(value)
            temp_byte = (temp_byte << 1) | 1
            count += 1
        else:
            temp_byte = (temp_byte << 1) | 0
            count += 1
        bit_count += 1
        if bit_count == 8:
            data_II.append(temp_byte)
            temp_byte = 0
            bit_count = 0
    if bit_count > 0:
        temp_byte <<= (8 - bit_count)
        data_II.append(temp_byte)
    return data_I, data_II, count

def jbit_decoding(data_I, data_II, original_length):
    """Decodifica J-bit Encoding per ripristinare i dati originali."""
    decoded_data = []
    bit_index = 0
    bit_count = 0
    for temp_byte in data_II:
        for i in range(7, -1, -1):
            if bit_count >= original_length:
                return decoded_data
            if (temp_byte >> i) & 1:
                if bit_index < len(data_I):
                    decoded_data.append(data_I[bit_index])
                    bit_index += 1
            else:
                decoded_data.append(0)
            bit_count += 1
    return decoded_data

# ============================
# FUNZIONI DI CODIFICA ARITMETICA CON RENORMALIZZAZIONE (ARITMETICA INTERA)
# ============================
def arithmetic_encode(symbols, precision=64):
    """
    Codifica la sequenza di simboli usando aritmetica intera con rinormalizzazione.
    Restituisce una tupla: (encoded_value, total_bits, ranges, length, total)
    """
    freq = Counter(symbols)
    total = sum(freq.values())
    # Costruiamo la tabella cumulativa ordinata per simbolo
    cum = {}
    cumulative = 0
    for sym in sorted(freq):
        cum[sym] = (cumulative, cumulative + freq[sym])
        cumulative += freq[sym]
    ranges = {sym: (low, high) for sym, (low, high) in cum.items()}
    
    low = 0
    high = (1 << precision) - 1
    HALF = 1 << (precision - 1)
    QUARTER = 1 << (precision - 2)
    THREE_QUARTER = 3 * QUARTER
    underflow = 0
    output_bits = []
    
    for symbol in symbols:
        range_width = high - low + 1
        sym_low, sym_high = cum[symbol]
        new_low = low + (range_width * sym_low) // total
        new_high = low + (range_width * sym_high) // total - 1
        low, high = new_low, new_high
        
        # Rinormalizzazione
        while True:
            if high < HALF:
                output_bits.append(0)
                for _ in range(underflow):
                    output_bits.append(1)
                underflow = 0
                low = low * 2
                high = high * 2 + 1
            elif low >= HALF:
                output_bits.append(1)
                for _ in range(underflow):
                    output_bits.append(0)
                underflow = 0
                low = (low - HALF) * 2
                high = (high - HALF) * 2 + 1
            elif low >= QUARTER and high < THREE_QUARTER:
                underflow += 1
                low = (low - QUARTER) * 2
                high = (high - QUARTER) * 2 + 1
            else:
                break

    underflow += 1
    if low < QUARTER:
        output_bits.append(0)
        for _ in range(underflow):
            output_bits.append(1)
    else:
        output_bits.append(1)
        for _ in range(underflow):
            output_bits.append(0)
    
    encoded_value_int = 0
    for bit in output_bits:
        encoded_value_int = (encoded_value_int << 1) | bit
    total_bits = len(output_bits)
    
    return encoded_value_int, total_bits, ranges, len(symbols), total

def arithmetic_decode(encoded_value_int, total_bits, ranges, length, total, precision=64):
    """
    Decodifica la codifica aritmetica ottenuta.
    Restituisce la lista dei simboli originali.
    """
    # Ricostruisco la tabella cumulativa a partire da ranges
    cum = {}
    for sym, (low_val, high_val) in ranges.items():
        cum[sym] = (low_val, high_val)
    
    low = 0
    high = (1 << precision) - 1
    HALF = 1 << (precision - 1)
    QUARTER = 1 << (precision - 2)
    THREE_QUARTER = 3 * QUARTER

    # Estraiamo il bitstream dall'intero encoded_value_int
    bitstream = []
    for i in range(total_bits):
        bit = (encoded_value_int >> (total_bits - i - 1)) & 1
        bitstream.append(bit)
    bits_iter = iter(bitstream)
    code = 0
    for _ in range(precision):
        try:
            bit = next(bits_iter)
        except StopIteration:
            bit = 0
        code = (code << 1) | bit
    
    decoded_symbols = []
    for _ in range(length):
        range_width = high - low + 1
        value = ((code - low + 1) * total - 1) // range_width
        for sym in sorted(cum.keys()):
            sym_low, sym_high = cum[sym]
            if sym_low <= value < sym_high:
                decoded_symbols.append(sym)
                new_low = low + (range_width * sym_low) // total
                new_high = low + (range_width * sym_high) // total - 1
                low, high = new_low, new_high
                break

        while True:
            if high < HALF:
                low = low * 2
                high = high * 2 + 1
                try:
                    bit = next(bits_iter)
                except StopIteration:
                    bit = 0
                code = (code * 2 + bit) & ((1 << precision) - 1)
            elif low >= HALF:
                low = (low - HALF) * 2
                high = (high - HALF) * 2 + 1
                try:
                    bit = next(bits_iter)
                except StopIteration:
                    bit = 0
                code = (code - HALF) * 2 + bit
            elif low >= QUARTER and high < THREE_QUARTER:
                low = (low - QUARTER) * 2
                high = (high - QUARTER) * 2 + 1
                try:
                    bit = next(bits_iter)
                except StopIteration:
                    bit = 0
                code = (code - QUARTER) * 2 + bit
            else:
                break

    return decoded_symbols

def parallel_arithmetic_encode(symbols, block_size=1024, precision=64):
    """Esegue codifica aritmetica in parallelo su blocchi."""
    blocks = [symbols[i:i+block_size] for i in range(0, len(symbols), block_size)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(arithmetic_encode, [(block, precision) for block in blocks])
    # Ogni risultato è una tupla (encoded_value, total_bits, ranges, length, total)
    encoded_values = [res[0] for res in results]
    total_bits_list = [res[1] for res in results]
    ranges_list = [res[2] for res in results]
    lengths = [res[3] for res in results]
    totals = [res[4] for res in results]
    return encoded_values, total_bits_list, ranges_list, lengths, totals

def parallel_arithmetic_decode(encoded_values, total_bits_list, ranges_list, lengths, totals, precision=64):
    """Decodifica aritmetica in parallelo per ciascun blocco e restituisce il flusso decodificato."""
    decoded_blocks = []
    for ev, tb, r, ln, tot in zip(encoded_values, total_bits_list, ranges_list, lengths, totals):
        decoded_block = arithmetic_decode(ev, tb, r, ln, tot, precision)
        decoded_blocks.extend(decoded_block)
    return decoded_blocks

# ============================
# SALVATAGGIO E CARICAMENTO FILE COMPRESSI
# ============================
def save_compressed_file(filename, original_length, 
                         encoded_data_I_blocks, encoded_data_II_blocks, alphabet):
    """
    Salva i dati compressi in un file binario.
    Vengono salvati:
      - Lunghezza originale (numero di bit della mappa J-bit)
      - L'alfabeto (usato per MTF)
      - Numero di blocchi per Data I, e per ciascun blocco:
          * encoded_value (intero convertito in stringa)
          * total_bits, total, length
          * Numero di simboli in ranges, e per ciascun simbolo: il byte, low e high
      - Lo stesso per Data II
    """
    with open(filename, "wb") as f:
        # Salvataggio metadati generali
        f.write(struct.pack("I", original_length))
        # Salvataggio alfabeto
        f.write(struct.pack("I", len(alphabet)))
        for char in alphabet:
            f.write(struct.pack("B", ord(char)))
        
        # Salvataggio blocchi per Data I
        num_blocks_I = len(encoded_data_I_blocks)
        f.write(struct.pack("I", num_blocks_I))
        for block in encoded_data_I_blocks:
            # Salva encoded_value come stringa
            encoded_str = str(block.encoded_value)
            encoded_str_len = len(encoded_str)
            f.write(struct.pack("I", encoded_str_len))
            f.write(encoded_str.encode('utf-8'))
            # Salva total_bits, total, length
            f.write(struct.pack("I", block.total_bits))
            f.write(struct.pack("I", block.total))
            f.write(struct.pack("I", block.length))
            # Salva ranges
            f.write(struct.pack("I", len(block.ranges)))
            for symbol, (low_val, high_val) in block.ranges.items():
                f.write(struct.pack("B", symbol))
                f.write(f"{low_val}\n".encode('utf-8'))
                f.write(f"{high_val}\n".encode('utf-8'))
        
        # Salvataggio blocchi per Data II
        num_blocks_II = len(encoded_data_II_blocks)
        f.write(struct.pack("I", num_blocks_II))
        for block in encoded_data_II_blocks:
            encoded_str = str(block.encoded_value)
            encoded_str_len = len(encoded_str)
            f.write(struct.pack("I", encoded_str_len))
            f.write(encoded_str.encode('utf-8'))
            f.write(struct.pack("I", block.total_bits))
            f.write(struct.pack("I", block.total))
            f.write(struct.pack("I", block.length))
            f.write(struct.pack("I", len(block.ranges)))
            for symbol, (low_val, high_val) in block.ranges.items():
                f.write(struct.pack("B", symbol))
                f.write(f"{low_val}\n".encode('utf-8'))
                f.write(f"{high_val}\n".encode('utf-8'))

def load_compressed_file(filename):
    """
    Carica i dati compressi da file binario e ricostruisce:
      - original_length
      - alfabeto
      - lista di blocchi per Data I (lista di EncodedData)
      - lista di blocchi per Data II (lista di EncodedData)
    """
    with open(filename, "rb") as f:
        original_length = struct.unpack("I", f.read(4))[0]
        alphabet_size = struct.unpack("I", f.read(4))[0]
        alphabet = "".join(chr(struct.unpack("B", f.read(1))[0]) for _ in range(alphabet_size))
        
        # Caricamento blocchi per Data I
        num_blocks_I = struct.unpack("I", f.read(4))[0]
        encoded_data_I_blocks = []
        for _ in range(num_blocks_I):
            encoded_str_len = struct.unpack("I", f.read(4))[0]
            encoded_value = int(f.read(encoded_str_len).decode('utf-8'))
            total_bits = struct.unpack("I", f.read(4))[0]
            total = struct.unpack("I", f.read(4))[0]
            length = struct.unpack("I", f.read(4))[0]
            num_symbols = struct.unpack("I", f.read(4))[0]
            ranges = {}
            for _ in range(num_symbols):
                symbol = struct.unpack("B", f.read(1))[0]
                low_val = int(Decimal(f.readline().decode().strip()))
                high_val = int(Decimal(f.readline().decode().strip()))
                ranges[symbol] = (low_val, high_val)
            encoded_data_I_blocks.append(EncodedData(encoded_value, total_bits, total, ranges, length))
        
        # Caricamento blocchi per Data II
        num_blocks_II = struct.unpack("I", f.read(4))[0]
        encoded_data_II_blocks = []
        for _ in range(num_blocks_II):
            encoded_str_len = struct.unpack("I", f.read(4))[0]
            encoded_value = int(f.read(encoded_str_len).decode('utf-8'))
            total_bits = struct.unpack("I", f.read(4))[0]
            total = struct.unpack("I", f.read(4))[0]
            length = struct.unpack("I", f.read(4))[0]
            num_symbols = struct.unpack("I", f.read(4))[0]
            ranges = {}
            for _ in range(num_symbols):
                symbol = struct.unpack("B", f.read(1))[0]
                low_val = int(Decimal(f.readline().decode().strip()))
                high_val = int(Decimal(f.readline().decode().strip()))
                ranges[symbol] = (low_val, high_val)
            encoded_data_II_blocks.append(EncodedData(encoded_value, total_bits, total, ranges, length))
    
    return original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet

# ============================
# FUNZIONI DI COMPRESSIONE E DECOMPRESSIONE
# ============================
def compress_file(input_text, output_file):
    """Esegue la compressione completa e salva il file."""
    # Step 1: Run-Length Encoding
    rle_output = run_length_encoding(input_text)
    rle_string = "".join(f"{char}{str(count)}" for char, count in rle_output)
    
    # Step 2: Burrows-Wheeler Transform
    bwt_output = parallel_bwt(rle_string)
    
    # Step 3: Creazione alfabeto per MTF
    alphabet = sorted(set(bwt_output))
    
    # Step 4: Move-To-Front Transform
    mtf_output = parallel_mtf(bwt_output)
    
    # Step 5: J-bit Encoding
    data_I, data_II, count_bits = jbit_encoding(mtf_output)
    
    # Step 6: Codifica Aritmetica in blocchi per Data I e Data II
    encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I = parallel_arithmetic_encode(data_I)
    encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II = parallel_arithmetic_encode(data_II)
    
    # Costruiamo le liste di blocchi
    encoded_data_I_blocks = [EncodedData(ev, tb, tot, r, ln)
                               for ev, tb, r, ln, tot in zip(encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I)]
    encoded_data_II_blocks = [EncodedData(ev, tb, tot, r, ln)
                                for ev, tb, r, ln, tot in zip(encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II)]
    
    save_compressed_file(output_file, count_bits, encoded_data_I_blocks, encoded_data_II_blocks, alphabet)

def decompress_file(input_file):
    """Esegue la decompressione dal file salvato e restituisce il testo originale."""
    original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet = load_compressed_file(input_file)
    alphabet = sorted(set(alphabet))
    
    # Decodifica aritmetica per ogni blocco e concatenazione dei risultati
    decoded_I = parallel_arithmetic_decode([block.encoded_value for block in encoded_data_I_blocks],
                                             [block.total_bits for block in encoded_data_I_blocks],
                                             [block.ranges for block in encoded_data_I_blocks],
                                             [block.length for block in encoded_data_I_blocks],
                                             [block.total for block in encoded_data_I_blocks])
    decoded_II = parallel_arithmetic_decode([block.encoded_value for block in encoded_data_II_blocks],
                                              [block.total_bits for block in encoded_data_II_blocks],
                                              [block.ranges for block in encoded_data_II_blocks],
                                              [block.length for block in encoded_data_II_blocks],
                                              [block.total for block in encoded_data_II_blocks])
    # Step 7: J-bit Decoding
    decoded_mtf = jbit_decoding(decoded_I, decoded_II, original_length)
    # Step 8: Move-To-Front Decoding
    decoded_bwt = move_to_front_decoding(decoded_mtf, alphabet)
    # Step 9: Inverse BWT
    rle_string = inverse_burrows_wheeler(decoded_bwt)
    # Step 10: Run-Length Decoding
    original_text = run_length_decoding(rle_string)
    return original_text

def read_file(filename):
    """
    Legge i dati da un file e restituisce il contenuto come stringa.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

# ============================
# ESEMPIO DI UTILIZZO
# ============================
if __name__ == "__main__":
    input_text = read_file("file1.txt")
    if input_text is None:
        print("Errore: file non trovato.")
        exit(1)
    compressed_file = "compressed.bin"
    compress_file(input_text, compressed_file)
    decompressed_text = decompress_file(compressed_file)
    
    # Verifica e scrittura del risultato
    if input_text == decompressed_text:
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("si")
    else:
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("no")
