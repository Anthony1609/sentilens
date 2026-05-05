"""
preprocessor.py
---------------
Text preprocessing pipeline (no NLTK dependency).
Handles lowercasing, URL/mention removal, stopword filtering, and Porter stemming.
"""

import re
import string

# ── Stopwords (curated English list) ──────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "not","no","nor","so","yet","both","either","neither","than","too","very",
    "just","also","this","that","these","those","i","me","my","we","our","you",
    "your","he","him","his","she","her","it","its","they","them","their","what",
    "which","who","whom","when","where","why","how","all","each","every","both",
    "few","more","most","other","some","such","up","out","about","into","then",
    "there","here","again","further","once","only","own","same","s","t","re",
    "ve","ll","d","m","didn","doesn","wasn","weren","hasn","haven","hadn",
    "isn","aren","won","wouldn","couldn","shouldn","mustn","needn",
}

# ── Minimal Porter Stemmer ─────────────────────────────────────────────────────
_STEP1A = [("sses","ss"),("ies","i"),("ss","ss"),("s","")]
_STEP1B_SUFFIXES = [("eed","ee"),("ed",""),("ing","")]
_STEP2 = [
    ("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
    ("izer","ize"),("bli","ble"),("alli","al"),("entli","ent"),("eli","e"),
    ("ousli","ous"),("ization","ize"),("ation","ate"),("ator","ate"),
    ("alism","al"),("iveness","ive"),("fulness","ful"),("ousness","ous"),
    ("aliti","al"),("iviti","ive"),("biliti","ble"),
]
_STEP3 = [
    ("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
    ("ical","ic"),("ful",""),("ness",""),
]
_STEP4 = [
    "al","ance","ence","er","ic","able","ible","ant","ement","ment",
    "ent","ion","ou","ism","ate","iti","ous","ive","ize",
]

def _count_vc(word):
    """Count vowel-consonant sequences (m measure)."""
    m, prev_vowel = 0, False
    for ch in word:
        v = ch in "aeiou"
        if v and not prev_vowel:
            pass
        elif not v and prev_vowel:
            m += 1
        prev_vowel = v
    return m

def _has_vowel(word):
    return any(c in "aeiou" for c in word)

def stem(word):
    """Simplified Porter Stemmer."""
    if len(word) <= 2:
        return word
    # Step 1a
    for suf, rep in _STEP1A:
        if word.endswith(suf):
            word = word[:-len(suf)] + rep
            break
    # Step 1b
    for suf, rep in _STEP1B_SUFFIXES:
        if word.endswith(suf):
            stem_part = word[:-len(suf)]
            if suf == "eed":
                if _count_vc(stem_part) > 0:
                    word = stem_part + rep
            else:
                if _has_vowel(stem_part):
                    word = stem_part + rep
                    if word.endswith(("at","bl","iz")):
                        word += "e"
                    elif len(word) > 1 and word[-1] == word[-2] and word[-1] not in "lsz":
                        word = word[:-1]
            break
    # Step 2
    for suf, rep in _STEP2:
        if word.endswith(suf):
            base = word[:-len(suf)]
            if _count_vc(base) > 0:
                word = base + rep
            break
    # Step 3
    for suf, rep in _STEP3:
        if word.endswith(suf):
            base = word[:-len(suf)]
            if _count_vc(base) > 0:
                word = base + rep
            break
    # Step 4
    for suf in _STEP4:
        if word.endswith(suf):
            base = word[:-len(suf)]
            if _count_vc(base) > 1:
                word = base
            break
    return word


# ── Main Preprocessor Class ────────────────────────────────────────────────────
class TextPreprocessor:
    """
    Preprocessing pipeline:
      1. Lowercase
      2. Expand common contractions
      3. Remove URLs, mentions, hashtag symbols, HTML tags
      4. Remove non-alphabetic characters
      5. Tokenize (whitespace split)
      6. Remove stopwords and short tokens
      7. Stem each token
    """

    _CONTRACTIONS = {
        "won't":"will not","can't":"cannot","n't":" not","'re":" are",
        "'ve":" have","'ll":" will","'d":" would","'m":" am",
        "it's":"it is","that's":"that is","there's":"there is",
        "they're":"they are","we're":"we are","you're":"you are",
        "i'm":"i am","isn't":"is not","aren't":"are not",
        "wasn't":"was not","weren't":"were not","haven't":"have not",
        "hasn't":"has not","hadn't":"had not","wouldn't":"would not",
        "couldn't":"could not","shouldn't":"should not","mustn't":"must not",
    }

    def preprocess(self, text: str) -> str:
        """Return a cleaned, stemmed string ready for vectorization."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        for k, v in self._CONTRACTIONS.items():
            text = text.replace(k, v)
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        # Remove @mentions and #hashtag symbols
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#", " ", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Keep only alphabetic characters
        text = re.sub(r"[^a-z\s]", " ", text)
        # Tokenize
        tokens = text.split()
        # Filter stopwords + short tokens, then stem
        tokens = [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
        return " ".join(tokens)

    def preprocess_batch(self, texts):
        return [self.preprocess(t) for t in texts]
