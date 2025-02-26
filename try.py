import numpy as np
import struct,os
from collections import Counter, namedtuple
from decimal import Decimal,getcontext
from fractions import Fraction
import multiprocessing
getcontext().prec = 100

EncodedData1 = namedtuple("EncodedData", ["encoded_value", "ranges","length"])

def run_length_encoding(data):
    """Applica Run-Length Encoding (RLE) a una stringa."""
    #print("\n[INFO] Eseguendo Run-Length Encoding (RLE)...")
    encoded = []
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            i += 1
            count += 1
        encoded.append((data[i], count))
        #print(f"    - Carattere: {data[i]}, Conteggio: {count}")
        i += 1
    return encoded
def run_length_decoding(encoded):
    """Decodifica Run-Length Encoding (RLE) dalla stringa."""
    decoded = []
    i = 0
    while i < len(encoded):
        char = encoded[i]  # Prende il carattere
        i += 1
        count_str = ""
        
        # Accumula tutti i numeri successivi come conteggio
        while i < len(encoded) and encoded[i].isdigit():
            count_str += encoded[i]
            i += 1
        
        count = int(count_str) if count_str else 1  # Converte il numero in intero
        
        decoded.extend([char] * count)  # Ricostruisce la sequenza
    return "".join(decoded)


def parallel_bwt(text, block_size=1024):
    """Applica BWT in parallelo su blocchi."""
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(burrows_wheeler_transform, blocks)  # Esegue BWT in parallelo
    
    return "".join(results)  # Ricombina i blocchi trasformati

def burrows_wheeler_transform(text, block_size=1024):
    """Applica BWT in blocchi per file di grandi dimensioni."""
    #print("\n[INFO] Eseguendo Burrows-Wheeler Transform (BWT)...")
    text += "$"
    #print(f"[DEBUG] Testo con delimitatore: {text}")
    
    if len(text) <= block_size:
        #print("[INFO] Testo più piccolo del block_size, applicazione diretta di BWT...")
        rotations = sorted([text[i:] + text[:i] for i in range(len(text))])
        #print("[DEBUG] Matrice delle rotazioni ordinata:")
        """
        for row in rotations:
            #print(f"  {row}")
        """
        result = "".join(row[-1] for row in rotations)
        #print(f"[INFO] Risultato BWT: {result}")
        return result
    
    #print(f"[INFO] Applicazione di BWT per blocchi di dimensione {block_size}...")
    output = []
    for i in range(0, len(text), block_size):
        block = text[i:i+block_size]
        #print(f"[DEBUG] Elaborazione blocco {i // block_size + 1}: {block}")
        rotations = sorted([block[j:] + block[:j] for j in range(len(block))])
        #print("[DEBUG] Matrice delle rotazioni per il blocco:")
        """
        for row in rotations:
            #print(f"  {row}")
        """
        transformed_block = "".join(row[-1] for row in rotations)
        #print(f"[INFO] BWT del blocco: {transformed_block}")
        output.append(transformed_block)
    
    final_result = "".join(output)
    #print(f"[INFO] Risultato finale BWT: {final_result}")
    return final_result    
def inverse_burrows_wheeler(bwt_string):
    """Ricostruisce la stringa originale dalla trasformata di Burrows-Wheeler."""
    table = ["" for _ in bwt_string]
    for _ in range(len(bwt_string)):
        table = sorted([bwt_string[i] + table[i] for i in range(len(bwt_string))])
    for row in table:
        if row.endswith("$"):
            return row[:-1]

def parallel_mtf(text, block_size=1024):
    """Esegue Move-To-Front Transform in parallelo su blocchi."""
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(move_to_front_transform, blocks)  # Esegue MTF in parallelo
    
    return [item for sublist in results for item in sublist]  # Unisce i risultati

def move_to_front_transform(text):
    """Applica Move-To-Front Transform (MTF) su una stringa."""
    #print("\n[INFO] Eseguendo Move-To-Front Transform (MTF)...")
    alphabet = sorted(set(text))  # Costruisce la lista ordinata dei caratteri unici
    #print("alfabeto",alphabet)
    encoded = []
    #print("Alfabeto iniziale",alphabet," basata su " + text)
    for char in text:
        index = alphabet.index(char)
        encoded.append(index)
        alphabet.insert(0, alphabet.pop(index))  # Sposta il carattere all'inizio della lista
        #print(f"    - Carattere: {char}, Indice: {index}, Lista aggiornata: {alphabet}")
    return encoded
def move_to_front_decoding(encoded,text):
    """Applica Move-To-Front Decoding (MTF)."""
    # Ricostruire l'alfabeto iniziale basato sui caratteri possibili
    #print("testo",text)
    alphabet = sorted(set(text))  # Deve corrispondere a quello della codifica
    #print(alphabet)
    decoded = []
    
    for index in encoded:
        symbol = alphabet[index]  # Trova il simbolo in base all'indice
        decoded.append(symbol)  # Aggiungi il simbolo alla sequenza decodificata
        alphabet.pop(index)  # Rimuovi il simbolo dall'alfabeto
        alphabet.insert(0, symbol)  # Sposta il simbolo in cima alla lista
    
    return "".join(decoded)  # Ricostruisce la stringa originale


def jbit_encoding(data):
    """Applica J-bit Encoding (JBE) separando dati e mappa di bit secondo il paper."""
    #print("\n[INFO] Eseguendo J-bit Encoding (JBE)...")
    data_I = []  # Byte diversi da zero
    data_II = []  # Mappa di bit
    temp_byte = 0  # Temporary byte per raccogliere bit
    bit_count = 0  # Contatore di bit nel temporary byte
    count = 0
    for value in data:
        if value != 0:
            data_I.append(value)
            temp_byte = (temp_byte << 1) | 1  # Aggiungere un '1' a data_II
            count +=1
        else:
            temp_byte = (temp_byte << 1) | 0  # Aggiungere uno '0'
            count +=1
        bit_count += 1
        #print(f"    - Valore: {value}, Temporary Byte: {bin(temp_byte)}, Bit Count: {bit_count}")
        
        if bit_count == 8:  # Quando il temporary byte è pieno, salvarlo in data_II
            #print("temp",temp_byte)
            data_II.append(temp_byte)
            temp_byte = 0
            bit_count = 0
    
    if bit_count > 0:  # Se rimangono bit non scritti, riempiamo fino a 8 bit
        temp_byte <<= (8 - bit_count)  # Aggiungiamo zeri nei bit finali
        data_II.append(temp_byte)
    
    #print("    - Dati I (Valori utili):", data_I)
    #print("    - Dati II (Mappa bit in binario):", [bin(x) for x in data_II])
    #print("    - Dati II (Mappa bit in binario):", " ".join(f"{bin(x)[2:].zfill(8)}" for x in data_II))
    
    return data_I, data_II,count






    """Applica Run-Length Encoding (RLE) a una stringa."""
    #print("\n[INFO] Eseguendo Run-Length Encoding (RLE)...")
    encoded = []
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            i += 1
            count += 1
        encoded.append((data[i], count))
        #print(f"    - Carattere: {data[i]}, Conteggio: {count}")
        i += 1
    return encoded

    """Applica la trasformazione di Burrows-Wheeler (BWT)."""
    #print("\n[INFO] Eseguendo Burrows-Wheeler Transform (BWT)...")
    text += "$"  # Aggiungere simbolo di fine stringa
    rotations = sorted([text[i:] + text[:i] for i in range(len(text))])
    """
    for i, row in enumerate(rotations):
        #print(f"    - Rotazione {i}: {row}")
    """
    bwt = "".join(row[-1] for row in rotations)
    #print("    - Risultato BWT:", bwt)
    return bwt
def jbit_decoding(data_I, data_II, original_length):
    """Applica la decodifica J-bit Encoding (JBE) per ripristinare l'input originale."""
    #print("qui",data_I,data_II)
    decoded_data = []
    bit_index = 0  # Indice per leggere `Data I`
    bit_count = 0  # Numero di bit letti
    #print(original_length)
    # Scansioniamo i byte di `Data II` (mappa bit)
    for temp_byte in data_II:
        #print(temp_byte)
        for i in range(7, -1, -1):  # Leggiamo ogni bit da sinistra a destra
            if bit_count >= original_length:  # Se abbiamo ricostruito l'input originale, fermiamo il processo
                return decoded_data
            #print("cc",temp_byte >> i & 1)
            if (temp_byte >> i) & 1:  # Se il bit è `1`, prendiamo un valore da `Data I`
                if bit_index < len(data_I):  # Controllo per evitare errori di accesso
                    decoded_data.append(data_I[bit_index])
                    bit_index += 1
            else:  # Se il bit è `0`, scriviamo uno zero byte
                decoded_data.append(0)
            
            bit_count += 1  # Aggiorniamo il conteggio dei bit usati

    return decoded_data

def parallel_arithmetic_encode(symbols, block_size=1024):
    """Esegue la codifica aritmetica in parallelo su blocchi."""
    blocks = [symbols[i:i+block_size] for i in range(0, len(symbols), block_size)]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(arithmetic_encode, blocks)  # Esegue encoding in parallelo
    
    # Dividere i risultati nelle tre componenti
    encoded_values = [res[0] for res in results]
    ranges_list = [res[1] for res in results]
    lengths = [res[2] for res in results]
    
    return encoded_values, ranges_list, lengths  # Restituisce i tre elementi separatamente


def arithmetic_encode(symbols):
    #print("\n[INFO] Eseguendo codifica aritmetica...")
    
    # Calcola la frequenza dei simboli
    freq = Counter(symbols)
    total = sum(freq.values())
    #print(f"[DEBUG] Frequenze dei simboli: {freq}")
    
    # Calcola gli intervalli cumulativi
    low = Decimal(0)
    high = Decimal(1)
    ranges = {}
    current_low = Decimal(0)
    
    for symbol, count in sorted(freq.items()):
        probability = Decimal(count) / Decimal(total)
        ranges[symbol] = (current_low, current_low + probability)
        #print(f"[DEBUG] Intervallo per {symbol}: {ranges[symbol]}")
        current_low += probability
    
    # Codifica aritmetica
    for symbol in symbols:
        low_range, high_range = ranges[symbol]
        range_width = high - low
        high = low + range_width * high_range
        low = low + range_width * low_range
        #print(f"[DEBUG] Simbolo: {symbol}, Low: {low}, High: {high}")
    
    # Il valore codificato può essere qualsiasi valore in (low, high)
    encoded_value = (low + high) / 2
    #print(f"[INFO] Valore codificato: {encoded_value}, Lunghezza simboli: {len(symbols)}")
    return encoded_value, ranges, len(symbols)
def arithmetic_decode(encoded_value, ranges, length):
    decoded_symbols = []
    low = Decimal(0)
    high = Decimal(1)
    # 🔥 Converti encoded_value in Fraction
    encoded_value = Decimal(encoded_value)
    for _ in range(length):
        value_range = high - low
        target_value = (encoded_value - low) / value_range
        
        for symbol, (range_low, range_high) in ranges.items():
            range_low = Decimal(range_low)  # 🔄 Converti in Fraction
            range_high = Decimal(range_high)  # 🔄 Converti in Fraction
            if Decimal(range_low) <= target_value < Decimal(range_high):
                decoded_symbols.append(symbol)
                high = low + value_range * Decimal(range_high)
                low = low + value_range * Decimal(range_low)
                break
    
    return decoded_symbols



def load_compressed_file(filename):
    """Carica i dati compressi da un file binario e ricostruisce gli oggetti EncodedData, inclusa la lettura dell'alfabeto iniziale."""
    with open(filename, "rb") as f:
        lenght = struct.unpack("I", f.read(4))[0]

        # 🔹 Legge l'alfabeto iniziale
        alphabet_size = struct.unpack("I", f.read(4))[0]
        alphabet = "".join(chr(struct.unpack("B", f.read(1))[0]) for _ in range(alphabet_size))

        # 🔥 Legge `encoded_value_I` come stringa e lo converte in Decimal
        encoded_str_len_I = struct.unpack("I", f.read(4))[0]
        encoded_value_I = Decimal(f.read(encoded_str_len_I).decode('utf-8'))

        # 🔥 Legge `encoded_value_II` come stringa e lo converte in Decimal
        encoded_str_len_II = struct.unpack("I", f.read(4))[0]
        encoded_value_II = Decimal(f.read(encoded_str_len_II).decode('utf-8'))

        # 🔹 Legge la lunghezza dei dati I e II
        length_I = struct.unpack("I", f.read(4))[0]
        num_symbols_I = struct.unpack("I", f.read(4))[0]
        ranges_I = {}
        for _ in range(num_symbols_I):
            symbol = struct.unpack("B", f.read(1))[0]
            low = Decimal(f.readline().decode().strip())  # 🔄 Converti stringa in Decimal
            high = Decimal(f.readline().decode().strip())  # 🔄 Converti stringa in Decimal
            ranges_I[symbol] = (low, high)

        encoded_data_I = EncodedData1(encoded_value_I, ranges_I, length_I)

        length_II = struct.unpack("I", f.read(4))[0]
        num_symbols_II = struct.unpack("I", f.read(4))[0]
        ranges_II = {}
        for _ in range(num_symbols_II):
            symbol = struct.unpack("B", f.read(1))[0]
            low = Decimal(f.readline().decode().strip())  # 🔄 Converti stringa in Decimal
            high = Decimal(f.readline().decode().strip())  # 🔄 Converti stringa in Decimal
            ranges_II[symbol] = (low, high)

        encoded_data_II = EncodedData1(encoded_value_II, ranges_II, length_II)

    return lenght, encoded_data_I, encoded_data_II, alphabet

def save_compressed_file(filename, original_length, encoded_data_I, encoded_data_II, alphabet):
    """Salva i dati compressi in un file binario, incluso l'alfabeto iniziale per la decodifica MTF."""
    with open(filename, "wb") as f:
        f.write(struct.pack("I", original_length))  # Lunghezza originale
        
        # 🔹 Scrive l'alfabeto iniziale per la decodifica MTF
        f.write(struct.pack("I", len(alphabet)))  # Numero di caratteri nell'alfabeto
        for char in alphabet:
            f.write(struct.pack("B", ord(char)))  # Salva ogni carattere come byte
        
        # 🔥 Salva `encoded_value_I` come stringa UTF-8
        encoded_str_I = str(encoded_data_I.encoded_value)
        encoded_str_len_I = len(encoded_str_I)
        f.write(struct.pack("I", encoded_str_len_I))  # Lunghezza della stringa
        f.write(encoded_str_I.encode('utf-8'))  # Scrive la stringa stessa
        
        # 🔥 Salva `encoded_value_II` come stringa UTF-8
        encoded_str_II = str(encoded_data_II.encoded_value)
        encoded_str_len_II = len(encoded_str_II)
        f.write(struct.pack("I", encoded_str_len_II))
        f.write(encoded_str_II.encode('utf-8'))

        # 🔹 Salva la lunghezza dei dati I e II
        f.write(struct.pack("I", encoded_data_I.length))
        f.write(struct.pack("I", len(encoded_data_I.ranges)))  # Numero di simboli in ranges_I
        for symbol, (low, high) in encoded_data_I.ranges.items():
            f.write(struct.pack("B", symbol))
            f.write(f"{low}\n".encode('utf-8'))   # Scrive come stringa per evitare perdita di precisione
            f.write(f"{high}\n".encode('utf-8'))  # Scrive come stringa per evitare perdita di precisione

        f.write(struct.pack("I", encoded_data_II.length))
        f.write(struct.pack("I", len(encoded_data_II.ranges)))  # Numero di simboli in ranges_II
        for symbol, (low, high) in encoded_data_II.ranges.items():
            f.write(struct.pack("B", symbol))
            f.write(f"{low}\n".encode('utf-8'))
            f.write(f"{high}\n".encode('utf-8'))

def compress_file(input_text, output_file):
    """Esegue la compressione completa e salva il file."""
    
    rle_output = run_length_encoding(input_text)
    rle_string = "".join(f"{char}{str(count)}" for char, count in rle_output)

    bwt_output = parallel_bwt(rle_string)
    
    # 🔹 Creiamo l'alfabeto prima della MTF
    alphabet = sorted(set(bwt_output))

    mtf_output = parallel_mtf(bwt_output)

    data_I, data_II, count = jbit_encoding(mtf_output)

    encoded_values_I, ranges_list_I, lengths_I = parallel_arithmetic_encode(data_I)
    encoded_values_II, ranges_list_II, lengths_II = parallel_arithmetic_encode(data_II)

    # 🔹 Prendiamo solo il primo valore per ogni blocco
    encoded_data1 = EncodedData1(encoded_values_I[0], ranges_list_I[0], lengths_I[0])
    encoded_data2 = EncodedData1(encoded_values_II[0], ranges_list_II[0], lengths_II[0])

    save_compressed_file(output_file, count, encoded_data1, encoded_data2, alphabet)


def decompress_file(input_file):
    """Esegue la decompressione dal file salvato e restituisce il testo originale."""
    #print("[INFO] Avvio decompressione...")

    # Step 1: Carica il file compresso
    lenght, encoded_I, encoded_II,alphabet = load_compressed_file(input_file)
    alphabet = sorted(set(alphabet))
    #print(lenght, encoded_I, encoded_II,alphabet)
    # Step 2: Decodifica Arithmetic Encoding
    decoded_I = arithmetic_decode(encoded_I.encoded_value, encoded_I.ranges,encoded_I.length)
    decoded_II = arithmetic_decode(encoded_II.encoded_value, encoded_II.ranges,encoded_II.length)
    #print("Dal aritmetica otteniamo il decode -> ",decoded_I,decoded_II)
    # Step 3: J-bit Decoding
    decoded_mtf = jbit_decoding(decoded_I, decoded_II, lenght)
    #print("Dato MTF otteniamo la sua decodifica",decoded_mtf)
    # Step 4: Move-To-Front Decoding
    decoded_bwt = move_to_front_decoding(decoded_mtf,alphabet)
    #print("Dio",decoded_bwt)
    # Step 5: Inverse Burrows-Wheeler Transform
    #print("qui",decoded_bwt)
    rle_string = inverse_burrows_wheeler(decoded_bwt)

    # Step 6: Run-Length Decoding
    #print(rle_string)
    original_text = run_length_decoding(rle_string)

    #print("[INFO] Decompressione completata.")
    return original_text or "gay"

def read_file(filename):
    """
    Legge i dati da un file e restituisce il contenuto come stringa.
    
    Args:
        filename (str): Il percorso del file da leggere.
        
    Returns:
        str: Il contenuto del file come stringa.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        #print(f"[ERRORE] Il file '{filename}' non è stato trovato.")
        return None
    except Exception as e:
        #print(f"[ERRORE] Si è verificato un errore durante la lettura del file: {e}")
        return None


# ============================
# 🚀 ESEMPIO DI UTILIZZO
# ============================
if __name__ == "__main__":
    

    #print("Directory corrente:", os.getcwd())
    input_text = read_file("file2.txt")
    compressed_file = "compressed.bin"
    compress_file(input_text,compressed_file)

    decompressed_text = decompress_file(compressed_file)

    # Verifica
    #print("\n[RISULTATO] Testo originale:", input_text)
    #print("[RISULTATO] Testo decompresso:", decompressed_text)

    """
    assert input_text == decompressed_text, "Errore! Il testo decompresso non corrisponde all'originale!"
    print("[✅] Compressione e decompressione riuscite con successo!")
    """
    if input_text == decompressed_text:
    # Scrive il testo decompresso in un file
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("si")
    else:
        with open("output.txt", "w", encoding="utf-8") as output_file:
            output_file.write("no")
