def get_type(s):
    """Determine if each suffix is L-type or S-type."""
    n = len(s)
    t = [0] * n
    t[-1] = 1  # Empty suffix is S-type
    for i in range(n-2, -1, -1):
        if s[i] > s[i+1] or (s[i] == s[i+1] and t[i+1] == 0):
            t[i] = 0  # L-type
        else:
            t[i] = 1  # S-type
    return t

def find_buckets(s):
    """Find the head and tail of each bucket."""
    buckets = {}
    for c in s:
        buckets[c] = buckets.get(c, 0) + 1
    
    # Calculate bucket heads and tails
    heads = {}
    tails = {}
    start = 0
    for c in sorted(buckets.keys()):
        heads[c] = start
        tails[c] = start + buckets[c] - 1
        start += buckets[c]
    return heads, tails

def induce_l(sa, s, t, heads):
    """Induce L-type suffixes."""
    for i in range(len(sa)):
        j = sa[i] - 1
        if j >= 0 and t[j] == 0:  # If j exists and is L-type
            c = s[j]
            sa[heads[c]] = j
            heads[c] += 1

def induce_s(sa, s, t, tails):
    """Induce S-type suffixes."""
    for i in range(len(sa)-1, -1, -1):
        j = sa[i] - 1
        if j >= 0 and t[j] == 1:  # If j exists and is S-type
            c = s[j]
            sa[tails[c]] = j
            tails[c] -= 1

def sais(s):
    """Construct suffix array using the SA-IS algorithm."""
    n = len(s)
    if n <= 1:
        return list(range(n))
    
    # Get types of each position
    t = get_type(s)
    
    # Find LMS positions
    lms = [i for i in range(1, n) if t[i] == 1 and t[i-1] == 0]
    
    # Initialize suffix array
    sa = [-1] * n
    
    # Get bucket boundaries
    heads, tails = find_buckets(s)
    
    # Place LMS suffixes
    for p in reversed(lms):
        c = s[p]
        sa[tails[c]] = p
        tails[c] -= 1
    
    # Induce L-type suffixes
    heads, _ = find_buckets(s)
    induce_l(sa, s, t, heads)
    
    # Induce S-type suffixes
    _, tails = find_buckets(s)
    induce_s(sa, s, t, tails)
    
    return sa

def bwt_transform(text):
    """Compute the Burrows-Wheeler Transform using SA-IS algorithm.
    This implementation is more memory efficient as it avoids generating
    all rotations explicitly."""
    # Add sentinel character if not present
    if not text.endswith('$'):
        text += '$'
    
    # Convert text to integer array for SA-IS
    int_text = [ord(c) for c in text]
    
    # Get suffix array
    sa = sais(int_text)
    
    # Compute BWT from suffix array
    n = len(text)
    bwt = [text[sa[i]-1] if sa[i] > 0 else '$' for i in range(n)]
    
    return ''.join(bwt)

def inverse_bwt(bwt):
    """Inverse Burrows-Wheeler Transform.
    This implementation uses less memory by avoiding the creation of the full matrix."""
    n = len(bwt)
    
    # Count occurrences of each character
    counts = {}
    for c in bwt:
        counts[c] = counts.get(c, 0) + 1
    
    # Calculate first column (sorted bwt)
    first_col = []
    running_total = 0
    char_to_range = {}
    for c in sorted(counts.keys()):
        char_to_range[c] = (running_total, running_total + counts[c])
        first_col.extend([c] * counts[c])
        running_total += counts[c]
    
    # Create last-to-first mapping
    char_counts = {c: 0 for c in counts}
    lf_mapping = [0] * n
    for i, c in enumerate(bwt):
        lf_mapping[i] = char_to_range[c][0] + char_counts[c]
        char_counts[c] += 1
    
    # Reconstruct original text
    result = [''] * n
    pos = 0  # Start from '$'
    for i in range(n-1, -1, -1):
        result[i] = bwt[pos]
        pos = lf_mapping[pos]
    
    # Remove sentinel character
    return ''.join(result).rstrip('$')