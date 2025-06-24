import struct, os, time, gc, sys, ctypes
from collections import Counter, namedtuple
from decimal import getcontext, Decimal
from fractions import Fraction
import multiprocessing

# Utilizziamo EncodedData per ciascun blocco
EncodedData = namedtuple("EncodedData", ["encoded_value", "total_bits", "total", "ranges", "length"])


def build_suffix_array(text):
    if "$" not in text:
        text += "$"
    n = len(text)
    T = (ctypes.c_ubyte * n)(* [ord(c) for c in text])
    SA = (ctypes.c_int * n)()
    ret = divsufsort.divsufsort(T, SA, n)
    if ret != 0:
        raise RuntimeError("divsufsort failed with code %d" % ret)
    return [SA[i] for i in range(n)]




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
# FUNZIONI DI BURROWS-WHEELER (BWT) CON SA-IS
# ============================
def sais(s):
    """
    Implementazione dell'algoritmo SA-IS.
    s: lista di interi (in cui l'elemento 0 è il carattere sentinella).
    Restituisce il suffix array come lista di indici.
    """
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Classificazione in tipi S e L
    t = [False] * n  # True per S-type, False per L-type
    t[-1] = True
    for i in range(n - 2, -1, -1):
        if s[i] < s[i + 1]:
            t[i] = True
        elif s[i] == s[i + 1]:
            t[i] = t[i + 1]

    # Indici LMS (Left-most S-type)
    lms = [i for i in range(1, n) if t[i] and not t[i - 1]]

    # Calcolo delle dimensioni dei bucket per ciascun carattere
    alphabet_size = max(s) + 1
    bucket_sizes = [0] * alphabet_size
    for ch in s:
        bucket_sizes[ch] += 1

    # Calcola gli estremi dei bucket
    bucket_start = [0] * alphabet_size
    bucket_end = [0] * alphabet_size
    sum_ = 0
    for i in range(alphabet_size):
        bucket_start[i] = sum_
        sum_ += bucket_sizes[i]
        bucket_end[i] = sum_ - 1

    def induce(lms_order):
        SA = [-1] * n
        # Passo 1: Posiziona gli indici LMS nei bucket (da destra)
        b_end = bucket_end.copy()
        for i in lms_order[::-1]:
            ch = s[i]
            SA[b_end[ch]] = i
            b_end[ch] -= 1

        # Passo 2: Induzione dei tipi L (da sinistra)
        b_start = bucket_start.copy()
        for i in range(n):
            j = SA[i] - 1
            if j >= 0 and not t[j]:
                ch = s[j]
                SA[b_start[ch]] = j
                b_start[ch] += 1

        # Passo 3: Induzione dei tipi S (da destra)
        b_end = bucket_end.copy()
        for i in range(n - 1, -1, -1):
            j = SA[i] - 1
            if j >= 0 and t[j]:
                ch = s[j]
                SA[b_end[ch]] = j
                b_end[ch] -= 1
        return SA

    # Primo induzione: usa l'ordine naturale degli LMS
    SA = induce(lms)

    # Assegna nomi alle sottostringhe LMS
    lms_names = [-1] * n
    new_name = 0
    prev = -1
    # Estrae l'ordine dei LMS ordinati nel SA
    lms_order = [i for i in SA if i in lms]
    for i in lms_order:
        if prev == -1:
            lms_names[i] = new_name
            prev = i
        else:
            diff = False
            for d in range(n):
                if i + d == n or prev + d == n:
                    if i + d == n and prev + d == n:
                        break
                    else:
                        diff = True
                        break
                if s[i + d] != s[prev + d] or t[i + d] != t[prev + d]:
                    diff = True
                    break
                if d > 0 and ((i + d in lms) or (prev + d in lms)):
                    break
            if diff:
                new_name += 1
            lms_names[i] = new_name
            prev = i

    new_seq = [lms_names[i] for i in lms if lms_names[i] != -1]

    if new_name + 1 == len(new_seq):
        new_lms_order = [0] * len(lms)
        for i, name in enumerate(new_seq):
            new_lms_order[name] = lms[i]
    else:
        new_lms_order = sais(new_seq)
        new_lms_order = [lms[i] for i in new_lms_order]

    SA = induce(new_lms_order)
    return SA

def suffix_array_sa_is(text):
    """
    Costruisce il suffix array della stringa 'text' usando SA-IS.
    La stringa viene convertita in una lista di interi e viene aggiunto un carattere sentinella (0).
    Restituisce il suffix array (senza l'indice del sentinella).
    """
    s = [ord(c) for c in text]
    s.append(0)  # carattere sentinella
    SA = sais(s)
    return SA[1:]

def bwt_sa_is(text):
    """
    Calcola la trasformazione di Burrows-Wheeler usando il suffix array ottenuto con SA-IS.
    Aggiunge il carattere terminatore '$' se non presente.
    """
    if "$" not in text:
        text += "$"
    SA = suffix_array_sa_is(text)
    n = len(text)
    bwt = [text[(i - 1) % n] for i in SA]
    return "".join(bwt)

# ... existing code ...

def burrows_wheeler_transform(text):
    """Applica la trasformazione di Burrows-Wheeler (BWT) usando SA-IS per una maggiore efficienza in memoria."""
    if len(text) > 1024 * 1024:  # For texts larger than 1MB
        blocks = []
        block_size = 1024 * 1024  # 1MB blocks
        for i in range(0, len(text), block_size):
            block = text[i:i+block_size]
            blocks.append(bwt_sa_is(block))
            print(f"BWT Progress: {(i+len(block))*100/len(text):.2f}%")
        return "".join(blocks)
    else:
        return bwt_sa_is(text)

def parallel_bwt(text):
    """Applica BWT senza parallelizzazione."""
    print("Starting BWT transformation...")
    return burrows_wheeler_transform(text)

# ... rest of the code ...
def inverse_burrows_wheeler(bwt_string):
    """Versione ottimizzata dell'inversione BWT."""
    if not bwt_string:
        return ""
        
    # Create first/last column mapping
    n = len(bwt_string)
    first_col = sorted(enumerate(bwt_string), key=lambda x: x[1])
    next_pos = [0] * n
    
    for i, (orig_pos, _) in enumerate(first_col):
        next_pos[orig_pos] = i
        
    result = []
    pos = next_pos[bwt_string.index('$')]
    for _ in range(n - 1):
        result.append(bwt_string[pos])
        pos = next_pos[pos]
        
    return ''.join(reversed(result))

# ============================
# FUNZIONI DI MOVE-TO-FRONT (MTF)
# ============================
def parallel_mtf(text, block_size=4096):
    """Move-To-Front Transform parallelo ottimizzato."""
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
    cum = {}
    for sym, (low_val, high_val) in ranges.items():
        cum[sym] = (low_val, high_val)
    
    low = 0
    high = (1 << precision) - 1
    HALF = 1 << (precision - 1)
    QUARTER = 1 << (precision - 2)
    THREE_QUARTER = 3 * QUARTER

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
        f.write(struct.pack("I", original_length))
        f.write(struct.pack("I", len(alphabet)))
        for char in alphabet:
            f.write(struct.pack("B", ord(char)))
        
        num_blocks_I = len(encoded_data_I_blocks)
        f.write(struct.pack("I", num_blocks_I))
        for block in encoded_data_I_blocks:
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
    print("\n=== COMPRESSION PHASE ===")
    original_size = len(input_text)
    
    print(f"Original size: {original_size} bytes")
    print(f"Original text: {input_text[:100]}..." if len(input_text) > 100 else input_text)
    print(f"Processing file of size: {len(input_text)} bytes")
    
    if len(input_text) > 1024 * 1024 * 100:
        print("Large file detected, using optimized processing...")
        gc.collect()
        
    # Step 1: Run-Length Encoding
    start_time = time.time()
    rle_output = run_length_encoding(input_text)
    rle_string = "".join(f"{char}{str(count)}" for char, count in rle_output)
    rle_time = time.time() - start_time
    print(f"\n1. Run-Length Encoding:")
    print(f"Output: {rle_string[:100]}..." if len(rle_string) > 100 else rle_string)
    print(f"Time: {rle_time:.4f} seconds")
    
    # Step 2: Burrows-Wheeler Transform (ottimizzato con SA-IS)
    start_time = time.time()
    bwt_output = parallel_bwt(rle_string)
    bwt_time = time.time() - start_time
    print(f"\n2. Burrows-Wheeler Transform:")
    print(f"Time: {bwt_time:.4f} seconds")
    
    # Step 3: Move-To-Front Transform
    alphabet = sorted(set(bwt_output))
    start_time = time.time()
    mtf_output = parallel_mtf(bwt_output)
    mtf_time = time.time() - start_time
    print(f"\n3. Move-To-Front Transform:")
    print(f"Time: {mtf_time:.4f} seconds")
    
    # Step 4: J-bit Encoding
    start_time = time.time()
    data_I, data_II, count_bits = jbit_encoding(mtf_output)
    jbe_time = time.time() - start_time
    print(f"\n4. J-bit Encoding:")
    print(f"Time: {jbe_time:.4f} seconds")
    
    # Step 5: Arithmetic Encoding
    start_time = time.time()
    encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I = parallel_arithmetic_encode(data_I)
    encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II = parallel_arithmetic_encode(data_II)
    arithmetic_time = time.time() - start_time
    print(f"\n5. Arithmetic Encoding:")
    print(f"Time: {arithmetic_time:.4f} seconds")

    total_time = rle_time + bwt_time + mtf_time + jbe_time + arithmetic_time
    print(f"\nTotal compression time: {total_time:.4f} seconds")
    
    encoded_data_I_blocks = [EncodedData(ev, tb, tot, r, ln)
                           for ev, tb, r, ln, tot in zip(encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I)]
    encoded_data_II_blocks = [EncodedData(ev, tb, tot, r, ln)
                            for ev, tb, r, ln, tot in zip(encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II)]
    
    save_compressed_file(output_file, count_bits, encoded_data_I_blocks, encoded_data_II_blocks, alphabet)
    compressed_size = os.path.getsize(output_file)
    compression_ratio = (compressed_size / original_size) * 100
    space_saving = 100 - compression_ratio
    
    print("\n=== COMPRESSION STATISTICS ===")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    print(f"Space saving: {space_saving:.2f}%")

def decompress_file(input_file):
    """Esegue la decompressione dal file salvato e restituisce il testo originale."""
    print("\n=== DECOMPRESSION PHASE ===")
    
    start_time = time.time()
    original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet = load_compressed_file(input_file)
    alphabet = sorted(set(alphabet))
    load_time = time.time() - start_time
    print(f"\n1. Load Compressed File:")
    print(f"Time: {load_time:.4f} seconds")
    
    start_time = time.time()
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
    arithmetic_time = time.time() - start_time
    print(f"\n2. Arithmetic Decoding:")
    print(f"Time: {arithmetic_time:.4f} seconds")
    
    start_time = time.time()
    decoded_mtf = jbit_decoding(decoded_I, decoded_II, original_length)
    jbe_time = time.time() - start_time
    print(f"\n3. J-bit Decoding:")
    print(f"Time: {jbe_time:.4f} seconds")
    
    start_time = time.time()
    decoded_bwt = move_to_front_decoding(decoded_mtf, alphabet)
    mtf_time = time.time() - start_time
    print(f"\n4. Move-To-Front Decoding:")
    print(f"Time: {mtf_time:.4f} seconds")
    
    start_time = time.time()
    rle_string = inverse_burrows_wheeler(decoded_bwt)
    bwt_time = time.time() - start_time
    print(f"\n5. Inverse Burrows-Wheeler Transform:")
    print(f"Output: {rle_string[:100]}..." if len(rle_string) > 100 else rle_string)
    print(f"Time: {bwt_time:.4f} seconds")
    
    start_time = time.time()
    original_text = run_length_decoding(rle_string)
    rle_time = time.time() - start_time
    print(f"\n6. Run-Length Decoding:")
    print(f"Final output: {original_text[:100]}..." if len(original_text) > 100 else original_text)
    print(f"Time: {rle_time:.4f} seconds")
    
    total_time = load_time + arithmetic_time + jbe_time + mtf_time + bwt_time + rle_time
    print(f"\nTotal decompression time: {total_time:.4f} seconds")
    
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
    nameFile = sys.argv[1]
    input_text = read_file(sys.argv[1])
    if input_text is None:
        print("Errore: file non trovato.")
        exit(1)
    input_size = os.path.getsize(nameFile)
    print("burrow", burrows_wheeler_transform("c1i1a1o1"))
    print("move", move_to_front_transform("1oica1$11"))
    print("inverse", inverse_burrows_wheeler("1oica1$11"))
    print(f"\nInput file size: {input_size:,} bytes")
    compressed_file = "compressed.bin"
    compress_file(input_text, compressed_file)
    decompressed_text = decompress_file(compressed_file)
    compressed_size = os.path.getsize(compressed_file)
    print("\n=== FINAL STATISTICS ===")
    print(f"Original file size: {input_size:,} bytes")
    print(f"Compressed file size: {compressed_size:,} bytes")
    print(f"Final compression ratio: {(compressed_size/input_size)*100:.2f}%")

    if input_text == decompressed_text:
        print("Successo")
    else:
        print("Insuccesso")
