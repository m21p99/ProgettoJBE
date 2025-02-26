def transform_bwt(input_str):
    """
    Implementazione dell'algoritmo Burrows-Wheeler Transform (BWT) in Python.
    Restituisce la colonna finale (BWT) e la posizione dell'input originale nella lista ordinata.
    """
    # Aggiungere il marcatore di fine stringa
    input_str += "$"
    
    # Generare tutte le rotazioni della stringa
    rotations = [input_str[i:] + input_str[:i] for i in range(len(input_str))]
    
    # Ordinare le rotazioni lessicograficamente
    sorted_rotations = sorted(rotations)
    
    # Estrarre l'ultima colonna delle rotazioni ordinate
    last_column = ''.join(rotation[-1] for rotation in sorted_rotations)
    
    # Trovare la posizione della stringa originale nella lista ordinata
    key = sorted_rotations.index(input_str)
    
    return last_column, key

def inverse_bwt(last_column):
    """
    Implementazione dell'algoritmo Inverse Burrows-Wheeler Transform (BWT) in Python.
    """
    n = len(last_column)
    
    # Creare la prima colonna ordinando la colonna finale
    first_column = sorted(last_column)
    
    # Creare un dizionario per tracciare le posizioni dei caratteri
    rank = {}
    for i, char in enumerate(first_column):
        if char not in rank:
            rank[char] = []
        rank[char].append(i)
    
    # Ricostruire la stringa originale
    result = []
    index = 0
    for _ in range(n):
        result.append(last_column[index])
        index = rank[result[-1]].pop(0)
    
    # Unire la stringa e rimuovere il marcatore di fine stringa '$'
    return ''.join(result).rstrip('$')





def transform_BWT(data):
    rotations = [data[i:] + data[:i] for i in range(len(data))]
    sorted_rotations = sorted(rotations)
    transformed_data = "".join(rotation[-1] for rotation in sorted_rotations)
    key = sorted_rotations.index(data)
    return transformed_data, key

def inverse_BWT(transformed_data, key):
    table = [''] * len(transformed_data)
    for i in range(len(transformed_data)):
        table = sorted([transformed_data[i] + table[i] for i in range(len(transformed_data))])
    original_data = table[key]
    return original_data

# Exemple 1
data = "A4B4C4"
transformed_data, key = transform_bwt(data)
#original_data = inverse_bwt(transformed_data)
print("Exemple 1:")
print("Données d'origine:", data)
print("Transformée de Burrows-Wheeler:", transformed_data)
#print("Données inversées:", original_data)