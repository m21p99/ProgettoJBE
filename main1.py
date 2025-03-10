import struct, os
from collections import Counter, namedtuple
import multiprocessing
from pysais import sais

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
# FUNZIONI DI BURROWS-WHEELER (BWT) OTTIMIZZATE
# ============================
def bwt_transform(text):
    """
    Costruisce la trasformata di Burrows-Wheeler usando un array dei suffissi costruito
    con pysais (algoritmo O(n)).
    """
    # Converte la stringa in una lista di interi (codici ASCII)
    int_text = [ord(c) for c in text]
    sa = sais(int_text)
    # Per ogni indice nell'array dei suffissi, se l'indice è 0 restituiamo il delimitatore
    return "".join(text[i-1] if i > 0 else "$" for i in sa)

# Se necessario, si può parallelizzare su blocchi anche questa fase,
# ma spesso la costruzione del SA ottimizzato è già molto più veloce.
def parallel_bwt(text, block_size=1024):
    """Esegue BWT in parallelo su blocchi usando la versione ottimizzata."""
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    with multiprocessing.Pool() as pool:
        results = pool.map(bwt_transform, blocks)
    return "".join(results)

def inverse_burrows_wheeler(bwt_string):
    """
    Ricostruisce la stringa originale dalla trasformata di Burrows-Wheeler
    usando un approccio iterativo (non ottimizzato, ma sufficiente per file moderati).
    """
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

    bitstream = [(encoded_value_int >> (total_bits - i - 1)) & 1 for i in range(total_bits)]
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
def save_compressed_file(filename, original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet):
    """
    Salva i dati compressi in un file binario.
    Vengono salvati:
      - Lunghezza originale (numero di bit della mappa J-bit)
      - L'alfabeto (usato per MTF)
      - Numero di blocchi per Data I e per Data II, con i relativi metadati.
    """
    with open(filename, "wb") as f:
        f.write(struct.pack("I", original_length))
        f.write(struct.pack("I", len(alphabet)))
        for char in alphabet:
            f.write(struct.pack("B", ord(char)))
        # Data I
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
                f.write(f"{low_val}\\n".encode('utf-8'))
                f.write(f"{high_val}\\n".encode('utf-8'))
        # Data II
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
                f.write(f"{low_val}\\n".encode('utf-8'))
                f.write(f"{high_val}\\n".encode('utf-8'))

def load_compressed_file(filename):
    """
    Carica i dati compressi da file binario e ricostruisce:
      - original_length, alfabeto e liste di blocchi per Data I e Data II.
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
                low_val = int(f.readline().decode().strip())
                high_val = int(f.readline().decode().strip())
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
                low_val = int(f.readline().decode().strip())
                high_val = int(f.readline().decode().strip())
                ranges[symbol] = (low_val, high_val)
            encoded_data_II_blocks.append(EncodedData(encoded_value, total_bits, total, ranges, length))
    return original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet

# ============================
# FUNZIONI DI COMPRESSIONE E DECOMPRESSIONE
# ============================
def compress_file(input_text, output_file):
    """Esegue la compressione completa e salva il file."""
    # 1. RLE
    rle_output = run_length_encoding(input_text)
    rle_string = "".join(f"{char}{count}" for char, count in rle_output)
    # 2. BWT (ottimizzato)
    bwt_output = bwt_transform(rle_string)
    # 3. Creazione alfabeto per MTF
    alphabet = sorted(set(bwt_output))
    # 4. MTF
    mtf_output = parallel_mtf(bwt_output)
    # 5. J-bit Encoding
    data_I, data_II, count_bits = jbit_encoding(mtf_output)
    # 6. Codifica Aritmetica
    encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I = parallel_arithmetic_encode(data_I)
    encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II = parallel_arithmetic_encode(data_II)
    encoded_data_I_blocks = [EncodedData(ev, tb, tot, r, ln) for ev, tb, r, ln, tot in zip(encoded_values_I, total_bits_list_I, ranges_list_I, lengths_I, totals_I)]
    encoded_data_II_blocks = [EncodedData(ev, tb, tot, r, ln) for ev, tb, r, ln, tot in zip(encoded_values_II, total_bits_list_II, ranges_list_II, lengths_II, totals_II)]
    save_compressed_file(output_file, count_bits, encoded_data_I_blocks, encoded_data_II_blocks, alphabet)

def decompress_file(input_file):
    """Esegue la decompressione dal file salvato e restituisce il testo originale."""
    original_length, encoded_data_I_blocks, encoded_data_II_blocks, alphabet = load_compressed_file(input_file)
    alphabet = sorted(set(alphabet))
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
    decoded_mtf = jbit_decoding(decoded_I, decoded_II, original_length)
    decoded_bwt = move_to_front_decoding(decoded_mtf, alphabet)
    rle_string = inverse_burrows_wheeler(decoded_bwt)
    original_text = run_length_decoding(rle_string)
    return original_text

# ============================
# MAIN (ESEMPIO DI UTILIZZO)
# ============================
if __name__ == "__main__":
    input_text = None
    try:
        with open("file1.txt", "r", encoding="utf-8") as file:
            input_text = file.read()
    except Exception as e:
        print("Errore: file non trovato o impossibile leggere il file.")
        exit(1)
        
    compressed_file = "compressed.bin"
    compress_file(input_text, compressed_file)
    decompressed_text = decompress_file(compressed_file)
    
    if input_text == decompressed_text:
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("si")
        print("Compressione e decompressione completate: risultato CORRETTO.")
    else:
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("no")
        print("Compressione e decompressione completate: risultato ERRATO.")
