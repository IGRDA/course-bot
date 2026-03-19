"""
Text preprocessing utilities for TTS optimization.

Prepares text for XTTS synthesis by:
- Splitting long sentences at natural break points
- Expanding acronyms that may have been missed by the LLM
- Normalizing numbers to their spoken form
- Cleaning problematic punctuation patterns
"""

import re
from typing import Optional


# Common acronyms and their spoken forms (language-specific)
ACRONYM_EXPANSIONS = {
    "en": {
        # Tech acronyms
        "API": "A.P.I.",
        "CPU": "C.P.U.",
        "GPU": "G.P.U.",
        "URL": "U.R.L.",
        "HTML": "H.T.M.L.",
        "CSS": "C.S.S.",
        "SQL": "S.Q.L.",
        "PDF": "P.D.F.",
        "USB": "U.S.B.",
        "RAM": "ram",
        "ROM": "rom",
        # Common abbreviations
        "vs": "versus",
        "etc": "etcetera",
        "ie": "that is",
        "eg": "for example",
        "PhD": "P.h" + ".D.",
        "CEO": "C.E.O.",
        "CTO": "C.T.O.",
        "CFO": "C.F.O.",
    },
    "es": {
        # Tech acronyms (same pronunciation hints)
        "API": "A.P.I.",
        "CPU": "C.P.U.",
        "GPU": "G.P.U.",
        "URL": "U.R.L.",
        "HTML": "H.T.M.L.",
        "CSS": "C.S.S.",
        "SQL": "S.Q.L.",
        "PDF": "P.D.F.",
        "USB": "U.S.B.",
        "RAM": "ram",
        "ROM": "rom",
        # Spanish-specific
        "PID": "P.I.D.",
        "IA": "I.A.",  # Inteligencia Artificial
        "OTAN": "O.T.A.N.",
        "ONU": "O.N.U.",
        "UE": "U.E.",
        "EEUU": "Estados Unidos",
        "etc": "etcétera",
    },
}

# Number words for basic normalization (catches what LLM missed)
NUMBER_WORDS_EN = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "one hundred",
}

NUMBER_WORDS_ES = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
    "10": "diez", "11": "once", "12": "doce", "13": "trece",
    "14": "catorce", "15": "quince", "16": "dieciséis", "17": "diecisiete",
    "18": "dieciocho", "19": "diecinueve", "20": "veinte", "30": "treinta",
    "40": "cuarenta", "50": "cincuenta", "60": "sesenta", "70": "setenta",
    "80": "ochenta", "90": "noventa", "100": "cien",
}


def split_long_sentences(text: str, max_words: int = 30) -> str:
    """Split sentences that exceed max_words at natural break points.
    
    Args:
        text: Input text to process
        max_words: Maximum words per sentence before splitting (default: 30)
        
    Returns:
        Text with long sentences split at natural break points
    """
    # Split into sentences (preserve the delimiter)
    sentence_pattern = r'([.!?]+\s*)'
    parts = re.split(sentence_pattern, text)
    
    result = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        delimiter = parts[i + 1] if i + 1 < len(parts) else ""
        
        words = sentence.split()
        
        if len(words) > max_words:
            # Find natural break points: commas, semicolons, conjunctions
            split_sentence = _split_at_natural_break(sentence, max_words)
            result.append(split_sentence)
        else:
            result.append(sentence)
        
        if delimiter:
            result.append(delimiter)
        
        i += 2 if delimiter else 1
    
    return "".join(result)


def _split_at_natural_break(sentence: str, max_words: int) -> str:
    """Split a long sentence at natural break points.
    
    Looks for:
    1. Commas with enough context on each side
    2. Conjunctions (and, but, or, because, which, that)
    3. Semicolons (convert to period)
    """
    words = sentence.split()
    
    # First, try splitting at semicolons (convert to period)
    if ";" in sentence:
        return sentence.replace(";", ".")
    
    # Find potential split points (commas, conjunctions)
    split_points = []
    word_count = 0
    char_pos = 0
    
    for i, word in enumerate(words):
        word_count += 1
        char_pos = sentence.find(word, char_pos)
        
        # Check for comma after this word
        next_char_pos = char_pos + len(word)
        if next_char_pos < len(sentence) and sentence[next_char_pos] == ",":
            # Good split point if we have enough words before and after
            words_remaining = len(words) - i - 1
            if word_count >= 10 and words_remaining >= 5:
                split_points.append((i, next_char_pos + 1, "comma"))
        
        # Check for conjunctions that could start a new sentence
        if word.lower() in ("and", "but", "or", "y", "pero", "o", "porque", "which", "que"):
            if word_count >= 12:  # Only split if we have substantial content before
                split_points.append((i, char_pos, "conjunction"))
        
        char_pos = next_char_pos
    
    # If no good split points, return as-is (TTS will handle it)
    if not split_points:
        return sentence
    
    # Use the best split point (prefer middle of sentence)
    target_pos = len(words) // 2
    best_split = min(split_points, key=lambda x: abs(x[0] - target_pos))
    
    split_idx, char_idx, split_type = best_split
    
    if split_type == "comma":
        # Convert comma to period, capitalize next word
        before = sentence[:char_idx].rstrip(",").strip()
        after = sentence[char_idx:].lstrip(", ")
        if after:
            after = after[0].upper() + after[1:]
        return f"{before}. {after}"
    else:
        # Conjunction: add period before it
        before = sentence[:char_idx].strip()
        after = sentence[char_idx:].strip()
        if after:
            after = after[0].upper() + after[1:]
        return f"{before}. {after}"


def expand_acronyms(text: str, language: str = "en") -> str:
    """Expand acronyms to their TTS-friendly spoken form.
    
    Uses language-specific expansion dictionaries. Acronyms are detected
    as all-caps words of 2-5 characters.
    
    Args:
        text: Input text
        language: Language code ("en", "es", etc.)
        
    Returns:
        Text with acronyms expanded for better TTS pronunciation
    """
    # Get language-specific expansions, fall back to English
    expansions = ACRONYM_EXPANSIONS.get(language, ACRONYM_EXPANSIONS.get("en", {}))
    
    def replace_acronym(match):
        word = match.group(0)
        # Check if it's in our expansion dictionary
        if word in expansions:
            return expansions[word]
        # For unknown acronyms, add periods between letters for letter-by-letter pronunciation
        # But only if it looks like an acronym (all caps, 2-5 chars)
        if word.isupper() and 2 <= len(word) <= 5:
            # Check if it's likely a word that should be spoken as-is
            # (e.g., "NASA", "SAPA" might be pronounced as words)
            # For now, leave unknown acronyms as-is - BPE will handle them
            return word
        return word
    
    # Match words that are all uppercase and 2-5 characters
    pattern = r'\b[A-Z]{2,5}\b'
    return re.sub(pattern, replace_acronym, text)


def normalize_numbers(text: str, language: str = "en") -> str:
    """Convert numeric values to their spoken word form.
    
    Handles:
    - Percentages (90% -> ninety percent)
    - Simple numbers (1-100)
    - Years in context
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        Text with numbers converted to words
    """
    number_words = NUMBER_WORDS_ES if language == "es" else NUMBER_WORDS_EN
    percent_word = "por ciento" if language == "es" else "percent"
    
    # Handle percentages first
    def replace_percent(match):
        num = match.group(1)
        if num in number_words:
            return f"{number_words[num]} {percent_word}"
        # For larger numbers, try to construct the word form
        return f"{_number_to_words(int(num), language)} {percent_word}"
    
    text = re.sub(r'(\d+)%', replace_percent, text)
    
    # Handle standalone numbers (be conservative - only small numbers)
    def replace_number(match):
        num = match.group(0)
        if num in number_words:
            return number_words[num]
        return num  # Leave complex numbers for TTS to handle
    
    # Only replace numbers that are standalone (not part of larger patterns)
    text = re.sub(r'\b(\d{1,2})\b(?![%\d,.])', replace_number, text)
    
    return text


def _number_to_words(n: int, language: str = "en") -> str:
    """Convert a number to its word representation (simple cases)."""
    number_words = NUMBER_WORDS_ES if language == "es" else NUMBER_WORDS_EN
    
    if str(n) in number_words:
        return number_words[str(n)]
    
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if language == "es":
            if tens == 20 and ones > 0:
                return f"veinti{number_words.get(str(ones), str(ones))}"
            connector = " y " if ones > 0 else ""
            ones_word = number_words.get(str(ones), str(ones)) if ones > 0 else ""
            return f"{number_words.get(str(tens), str(tens))}{connector}{ones_word}"
        else:
            connector = "-" if ones > 0 else ""
            ones_word = number_words.get(str(ones), str(ones)) if ones > 0 else ""
            return f"{number_words.get(str(tens), str(tens))}{connector}{ones_word}"
    
    # For larger numbers, return as-is
    return str(n)


def clean_punctuation(text: str) -> str:
    """Clean problematic punctuation patterns for TTS.
    
    Fixes:
    - Stacked punctuation (?!, ...?, !!)
    - Multiple consecutive punctuation marks
    - Semicolons (convert to periods)
    
    Args:
        text: Input text
        
    Returns:
        Text with cleaned punctuation
    """
    # Replace semicolons with periods and capitalize next word
    def capitalize_after_semicolon(match):
        return ". " + match.group(1).upper()
    
    text = re.sub(r';\s*([a-záéíóúüñ])', capitalize_after_semicolon, text, flags=re.IGNORECASE)
    
    # Remove stacked punctuation - keep only the first meaningful one
    # ?! -> ?
    # !? -> !
    # !! -> !
    # ...? -> ...
    # ?... -> ?
    text = re.sub(r'\?!+', '?', text)
    text = re.sub(r'!\?+', '!', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{3,}\?', '...', text)
    text = re.sub(r'\?\.{2,}', '?', text)
    text = re.sub(r'\.{4,}', '...', text)  # Normalize long ellipsis
    
    # Clean up double periods (can happen after acronym expansion)
    text = re.sub(r'\.{2}', '.', text)
    
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    return text.strip()


def preprocess_for_tts(text: str, language: str = "en") -> str:
    """Main entry point for TTS text preprocessing.
    
    Applies all preprocessing steps in order:
    1. Clean punctuation
    2. Expand acronyms
    3. Normalize numbers
    4. Split long sentences
    5. Final cleanup (double periods from acronym expansion)
    
    Args:
        text: Raw text to preprocess
        language: Language code ("en", "es", or ISO code for multilingual)
        
    Returns:
        Preprocessed text optimized for TTS synthesis
    """
    # Normalize language code (handle full language names)
    lang = language.lower()[:2] if language else "en"
    
    # Apply preprocessing steps
    text = clean_punctuation(text)
    text = expand_acronyms(text, lang)
    text = normalize_numbers(text, lang)
    text = split_long_sentences(text, max_words=30)
    
    # Final cleanup: fix double periods created by acronym expansion at sentence boundaries
    text = re.sub(r'\.{2,}', '.', text)
    
    return text

