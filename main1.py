#!/usr/bin/env python
import struct, os, time, gc, sys, multiprocessing, platform, ctypes, traceback, psutil
from collections import Counter, namedtuple
from decimal import getcontext, Decimal
from fractions import Fraction
from ctypes import *



# Named tuple per i blocchi compressi
EncodedData = namedtuple("EncodedData", ["encoded_value", "total_bits", "total", "ranges", "length"])

# ============================
# BINDING PER LIBDIVSUFSORT (in C)
# ============================
dll_path = os.path.join(r"C:\Users\mario\PycharmProjects\ProgettoJBE\libdivsufsort\build\examples\Release", "divsufsort.dll")
divsufsort = ctypes.CDLL(dll_path)
divsufsort.divsufsort.argtypes = [POINTER(ctypes.c_ubyte), POINTER(ctypes.c_int), ctypes.c_int]
divsufsort.divsufsort.restype = ctypes.c_int

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

def bwt_divsufsort(text):
    if "$" not in text:
        text += "$"
    SA = build_suffix_array(text)
    n = len(text)
    # Costruisce la BWT: per ogni indice i in SA, usa il carattere precedente (mod n)
    bwt = [text[(i - 1) % n] for i in SA]
    # (Opzionale: puoi salvare la posizione originale se necessario)
    return "".join(bwt), SA.index(0)

# ============================
# BINDING PER RLE (in C)
# ============================
# Specifica il percorso della DLL RLE (assicurati che il file rle.dll sia nel percorso indicato)
rle_dll_path = os.path.join(r"C:\Users\mario\PycharmProjects\ProgettoJBE\rle", "rle.dll")
rle_lib = ctypes.CDLL(rle_dll_path)

# Definizione delle firme delle funzioni in C:
# char* rle_encode(const char* input);
# Modifica le definizioni delle funzioni RLE
rle_lib.rle_encode.argtypes = [ctypes.c_char_p]
rle_lib.rle_encode.restype = ctypes.c_void_p  # Cambiato da c_char_p a c_void_p

rle_lib.rle_decode.argtypes = [ctypes.c_char_p]
rle_lib.rle_decode.restype = ctypes.c_void_p  # Cambiato da c_char_p a c_void_p

rle_lib.free_result.argtypes = [ctypes.c_void_p]  # Cambiato da c_char_p a c_void_p
rle_lib.free_result.restype = None


inverse_dll_path = os.path.join(r"C:\Users\mario\PycharmProjects\ProgettoJBE\bwt", "inverse_bwt.dll")
inverse_bwt_lib = ctypes.CDLL(inverse_dll_path)

inverse_bwt_lib.inverse_bwt.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
inverse_bwt_lib.inverse_bwt.restype = ctypes.c_char_p


def rle_encode_c(input_str: str) -> str:
    if not input_str:
        return ""
    input_bytes = input_str.encode('latin-1')
    print(f"Input bytes: {input_bytes}")  # Debug
    result_ptr = rle_lib.rle_encode(input_bytes)
    if not result_ptr:
        raise RuntimeError("RLE encoding failed")
    try:
        rle_string = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('latin-1')
        print(f"RLE encoded (first 100 chars): {rle_string[:100]}...")
    finally:
        rle_lib.free_result(result_ptr)
    return rle_string

def rle_decode_c(encoded_str: str) -> str:
    if not encoded_str:
        return ""
    encoded_bytes = encoded_str.encode('latin-1')
    result_ptr = rle_lib.rle_decode(encoded_bytes)
    if result_ptr:
        decoded_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('latin-1')
        rle_lib.free_result(result_ptr)
        return decoded_str.encode('latin-1').decode('utf-8')
    return ""

# ============================
# FUNZIONI DI MOVE-TO-FRONT (MTF)
# ============================

# ============================
# FUNZIONI DI BURROWS-WHEELER TRANSFORM
# ============================

"""
def parallel_bwt(text, block_size=999999999):
    # For small inputs, process as single block
    if len(text) < block_size:
        bwt_result, _ = bwt_divsufsort(text)
        return bwt_result
    
    # For larger texts, split into blocks and ensure proper block size
    block_size = min(block_size, 1024 * 1024)  # Max 1MB per block
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    
    # Use multiprocessing with a limited number of processes
    num_processes = min(multiprocessing.cpu_count(), 4)  # Limit to 4 processes
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(bwt_worker, blocks)
    
    return "".join(results)
"""
def parallel_bwt(text):
    """Parallel implementation of Burrows-Wheeler Transform with automatic block sizing"""
    # Calculate optimal block size based on system memory and text length
    available_memory = psutil.virtual_memory().available
    optimal_block_size = min(available_memory // 4, len(text))  # Use 25% of available memory
    block_size = max(1024 * 1024, optimal_block_size)  # Minimum 1MB block
    
    if len(text) <= block_size:
        bwt_result, _ = bwt_divsufsort(text)
        return bwt_result
    
    # Create overlapping blocks
    overlap = 1024  # Fixed overlap size
    blocks = []
    for i in range(0, len(text), block_size - overlap):
        end = min(i + block_size, len(text))
        block = text[max(0, i-overlap):end]
        blocks.append(block)
    
    num_processes = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(bwt_worker, blocks)
    
    # Merge blocks, removing overlapping regions
    final_result = []
    for i, result in enumerate(results):
        if i == 0:
            final_result.append(result)
        else:
            final_result.append(result[overlap:])
    
    return "".join(final_result)



def bwt_worker(text):
    # Add error handling for worker process
    try:
        result, _ = bwt_divsufsort(text)
        return result
    except Exception as e:
        print(f"Error in BWT worker: {e}")
        return text  # Return original text on error

"""
def inverse_burrows_wheeler(bwt_string):
    
    if not bwt_string:
        return ""
        
    # Add $ if not present
    if '$' not in bwt_string:
        bwt_string += '$'
        
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
        
    # Remove the $ if we added it
    result_str = ''.join(reversed(result))
    if result_str.endswith('$'):
        result_str = result_str[:-1]
    
    return result_str
"""

def inverse_bwt_c(bwt_string):
    """
    Inverte la BWT su un blocco utilizzando la funzione implementata in C.
    bwt_string: stringa BWT (in latin-1)
    Restituisce la stringa originale.
    """
    n = len(bwt_string)
    # Converti la stringa in bytes (usando latin-1 per una mappatura one-to-one)
    bwt_bytes = bwt_string.encode('latin-1')
    result_ptr = inverse_bwt_lib.inverse_bwt(bwt_bytes, n)
    if not result_ptr:
        raise RuntimeError("inverse_bwt in C ha fallito")
    # Ottieni la stringa risultante (assumendo che sia in latin-1)
    original = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('latin-1')
    # Se la libreria C alloca memoria con malloc, potresti doverla liberare qui
    # usando una funzione free_result esposta dalla DLL.
    return original
def move_to_front_transform(text):
    alphabet = list(sorted(set(text)))
    alphabet_dict = {c: i for i, c in enumerate(alphabet)}
    encoded = []
    for char in text:
        index = alphabet_dict[char]
        encoded.append(index)
        if index > 0:
            char = alphabet.pop(index)
            alphabet.insert(0, char)
            for i in range(index + 1):
                alphabet_dict[alphabet[i]] = i
    return encoded
"""
def parallel_mtf(text, block_size=999999999):
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    with multiprocessing.Pool() as pool:
        results = pool.map(move_to_front_transform, blocks)
    return [item for sublist in results for item in sublist]
"""
def parallel_mtf(text):
    """Parallel MTF with state preservation"""
    # Calculate optimal block size
    available_memory = psutil.virtual_memory().available
    optimal_block_size = min(available_memory // 4, len(text))
    block_size = max(1024 * 1024, optimal_block_size)
    
    if len(text) <= block_size:
        return move_to_front_transform(text)
    
    # Initialize global state
    global_alphabet = list(sorted(set(text)))
    blocks = []
    mtf_states = []
    current_state = global_alphabet.copy()
    
    # Process blocks with state preservation
    for i in range(0, len(text), block_size):
        block = text[i:i+block_size]
        blocks.append((block, current_state.copy()))
        
        # Update state for next block
        for char in block:
            if char in current_state:
                idx = current_state.index(char)
                if idx > 0:
                    current_state.pop(idx)
                    current_state.insert(0, char)
    
    # Process blocks in parallel with their respective states
    with multiprocessing.Pool() as pool:
        results = pool.starmap(mtf_transform_with_state, blocks)
    
    return [val for block_result in results for val in block_result]

def mtf_transform_with_state(text, initial_state):
    """MTF transform with initial state"""
    alphabet = initial_state.copy()
    encoded = []
    
    for char in text:
        index = alphabet.index(char)
        encoded.append(index)
        if index > 0:
            char = alphabet.pop(index)
            alphabet.insert(0, char)
    
    return encoded

def move_to_front_decoding(encoded, alphabet):
    # L'alfabeto deve essere una lista mutabile
    alphabet = list(alphabet)
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
    data_I = []
    data_II = []
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
# FUNZIONI DI ARITHMETIC CODING
# ============================
def arithmetic_encode(symbols, precision=64):
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
                low *= 2
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
        # Fine per questo simbolo
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
                low *= 2
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
    decoded_blocks = []
    for ev, tb, r, ln, tot in zip(encoded_values, total_bits_list, ranges_list, lengths, totals):
        decoded_block = arithmetic_decode(ev, tb, r, ln, tot, precision)
        decoded_blocks.extend(decoded_block)
    return decoded_blocks

# ============================
# FUNZIONI DI SALVATAGGIO/ CARICAMENTO FILE
# ============================
def save_compressed_file(filename, original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet):
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
    print("\n=== COMPRESSION PHASE ===")
    original_size = len(input_text)
    print(f"Original size: {original_size} bytes")
    print(f"Processing file of size: {len(input_text)} bytes")
    if len(input_text) > 1024 * 1024 * 100:
        print("Large file detected, using optimized processing...")
        gc.collect()
    
    # RLE Encoding tramite C
    start_time = time.time()
    input_bytes = input_text.encode('utf-8')
    result_ptr = rle_lib.rle_encode(input_bytes)
    if not result_ptr:
        raise RuntimeError("RLE encoding failed")
    
    rle_string = None
    try:
        # Usa string_at invece di cast diretto
        result = ctypes.string_at(result_ptr)
        rle_string = result.decode('latin-1')
        #print(f"RLE encoded length: {len(rle_string)}")
    finally:
        if result_ptr:
            rle_lib.free_result(result_ptr)
            print("Freed RLE memory")

    if not rle_string:
        raise RuntimeError("RLE encoding produced empty result")
    
    rle_time = time.time() - start_time
    print(f"\n1. RLE Encoding (C) completed:")
    print(f"Time: {rle_time:.4f} seconds")
    #print(f"Output RLE: {rle_string}")
    
    start_time = time.time()
    bwt_output = parallel_bwt(rle_string)
    bwt_time = time.time() - start_time
    print(f"\n2. Burrows-Wheeler Transform:")
    print(f"Time: {bwt_time:.4f} seconds")
    #print(f"Output BWT: {bwt_output}")
    
    # Verifica che l'output BWT non sia vuoto
    if not bwt_output:
        raise RuntimeError("BWT produced empty result")
    
    
    alphabet = sorted(set(bwt_output))
    start_time = time.time()
    mtf_output = parallel_mtf(bwt_output)
    mtf_time = time.time() - start_time
    print(f"\n3. Move-To-Font Transform:")
    print(f"Time: {mtf_time:.4f} seconds")
    #print(f"Output mtf: {mtf_output}")

    # Verifica che l'output MTF non sia vuoto
    if not mtf_output:
        raise RuntimeError("MTF produced empty result")
    

    start_time = time.time()
    data_I, data_II, count_bits = jbit_encoding(mtf_output)
    jbe_time = time.time() - start_time
    print(f"\n4. J-bit Encoding:")
    print(f"Time: {jbe_time:.4f} seconds")
    
    # Verifica che i dati J-bit non siano vuoti
    if not data_I or not data_II:
        raise RuntimeError("J-bit encoding produced empty result")
    
    
    start_time = time.time()
    encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I = parallel_arithmetic_encode(data_I)
    encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II = parallel_arithmetic_encode(data_II)
    arithmetic_time = time.time() - start_time
    print(f"\n5. Arithmetic Encoding:")
    print(f"Time: {arithmetic_time:.4f} seconds")
    
    total_time = rle_time + bwt_time + mtf_time + jbe_time + arithmetic_time
    print(f"\nTotal compression time: {total_time:.4f} seconds")
    
     # Verifica che i dati aritmetici non siano vuoti
    if not encoded_values_I or not encoded_values_II:
        raise RuntimeError("Arithmetic encoding produced empty result")
    

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
    print("\n=== DECOMPRESSION PHASE ===")
    start_time = time.time()
    original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet = load_compressed_file(input_file)
    alphabet = sorted(set(alphabet))
    load_time = time.time() - start_time
    print(f"\n1. Load Compressed File:")
    print(f"Time: {load_time:.4f} seconds")
    
    start_time = time.time()
    decoded_I = parallel_arithmetic_decode(
        [block.encoded_value for block in encoded_data_I_blocks],
        [block.total_bits for block in encoded_data_I_blocks],
        [block.ranges for block in encoded_data_I_blocks],
        [block.length for block in encoded_data_I_blocks],
        [block.total for block in encoded_data_I_blocks]
    )
    decoded_II = parallel_arithmetic_decode(
        [block.encoded_value for block in encoded_data_II_blocks],
        [block.total_bits for block in encoded_data_II_blocks],
        [block.ranges for block in encoded_data_II_blocks],
        [block.length for block in encoded_data_II_blocks],
        [block.total for block in encoded_data_II_blocks]
    )
    arithmetic_time = time.time() - start_time
    print(f"\n2. Arithmetic Decoding:")
    print(f"Time: {arithmetic_time:.4f} seconds")
    #print(f"input file jbit: {decoded_I, decoded_II, original_length}")
    start_time = time.time()
    decoded_mtf = jbit_decoding(decoded_I, decoded_II, original_length)
    jbe_time = time.time() - start_time
    print(f"\n3. J-bit Decoding:")
    print(f"Time: {jbe_time:.4f} seconds")
    
    start_time = time.time()
    decoded_bwt = move_to_front_decoding(decoded_mtf, alphabet)
    mtf_time = time.time() - start_time
    print(f"\n4. Move-To-Font Decoding:")
    print(f"Time: {mtf_time:.4f} seconds")
    #print(f"input file move to front: {decoded_mtf}")
    #print(f"input file move to front fase btw: {decoded_bwt}")
    
    start_time = time.time()
    rle_string = inverse_bwt_c(decoded_bwt)
    #print(f"input file rle: {rle_string}")
    bwt_time = time.time() - start_time
    print(f"\n5. Inverse BWT:")
    print(f"Time: {bwt_time:.4f} seconds")
    # Usa il binding C per RLE Decoding
    start_time = time.time()
    original_text = rle_decode_c(rle_string)
    #print(f"original text: {original_text}")
    rle_time = time.time() - start_time
    print(f"\n6. RLE Decoding (C):")
    print(f"Time: {rle_time:.4f} seconds")
    
    total_time = load_time + arithmetic_time + jbe_time + mtf_time + bwt_time + rle_time
    print(f"\nTotal decompression time: {total_time:.4f} seconds")
    return original_text

def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return None

if __name__ == "__main__":
    nameFile = sys.argv[1]
    input_text = read_file(sys.argv[1])
    if input_text is None:
        print("Errore: file non trovato.")
        exit(1)
    input_size = os.path.getsize(nameFile)
    print(f"\nInput file size: {input_size:,} bytes")
    compressed_file = "compressed.bin"
    #compress_file(input_text, compressed_file)
    try:
        compress_file(input_text, compressed_file)
    except Exception as e:
        print(f"Errore durante la compressione: {e}")
        traceback.print_exc()
    
    decompressed_text = decompress_file(compressed_file)
    
    if input_text == decompressed_text:
        print("\nSuccessful decompression!")
    else:
        print("\nDecompression mismatch!")
        
