import re
def are_words_in_same_parentheses(text, word1, word2):
    matches = re.findall(r'\(([^)]*)\)', text)
    for match in matches:
        if re.search(rf'\b{re.escape(word1)}\b', match) and re.search(rf'\b{re.escape(word2)}\b', match):
            return True
    return False
def is_followed_by_examples_containing(text, word1, word2):
    pattern = rf'''
        \b{re.escape(word1)}\b
        \s*
        \(\s*
        (?:e\.?g\.?|i\.?e\.?|such\s+as)
        [^)]*\b{re.escape(word2)}\b[^)]*
        \)
    '''
    return re.search(pattern, text, flags=re.IGNORECASE | re.VERBOSE) is not None
xpath_pair_query = "//pair"

def is_word_in_following_brackets(text, word1, word2):
    pattern = rf"""
        \b{re.escape(word1)}\b                    # Exact word1
        \s*                                       # Optional space
        (?P<bracket>\(|\[)                        # Capture opening bracket
        (?P<content>[^\)\]]*?)                    # Capture everything until closing bracket
        (\)|\])                                   # Closing bracket
    """

    for match in re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE):
        content = match.group("content").lower()
        candidates = [c.strip() for c in content.split(",")]
        if word2.lower() in candidates:
            return True

    return False


def is_word_followed_by_list(text, word1, word2):
    # Construct the pattern to match word1 followed by a list containing word2
    pattern = r"""
        \b{0}\b                       # Match word1 (e.g., DRUG0)
        \s*(?:[\(\[\{{]|,)?\s*       # Optional separator: :, (, [, {{, or comma
        (?P<list>                     # Start of the named group 'list'
            DRUG\d+                   # Match a drug
            (?:\s*(?:,|\+|or|and|OR|AND)\s*DRUG\d+)*  # Followed by other drugs with separators
        )
        \s*[\)\]\}}\.]?               # Optional closing: ), ], }}, .
    """.format(re.escape(word1))

    matches = re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE)
    for match in matches:
        list_part = match.group("list")
        # Normalize and split the list part into individual items
        items = re.split(r"\s*(?:,|\+|or|and|OR|AND)\s*", list_part)
        cleaned = [item.strip().lower() for item in items if item.strip().lower().startswith("drug")]
        if word2.lower() in cleaned:
            return True
    return False

def is_word_followed_by_comma_and_or_plus(text, word1, word2):
    # Construct the pattern to match word1 followed by a comma and optional and/or/+
    pattern = r"""
        \b{0}\b                       # Match word1 (e.g., DRUG2)
        \s*,\s*(?:(?:and|or|\+))?\s*  # Comma with optional 'and', 'or', or '+'
        (?P<list>                     # Start of the named group 'list'
            DRUG\d+                   # Match a drug
            (?:\s*,\s*(?:(?:and|or|\+))?\s*DRUG\d+)*  # Optional additional drugs with comma and optional and/or/+
        )
        \s*\.?                        # Optional closing period
    """.format(re.escape(word1))

    matches = re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE)
    for match in matches:
        list_part = match.group("list")
        # Normalize and split the list part into individual items
        items = re.split(r"\s*,\s*(?:(?:and|or|\+))?\s*", list_part)
        cleaned = [item.strip().lower() for item in items if item.strip().lower().startswith("drug")]
        if word2.lower() in cleaned:
            return True
    return False

def is_words_between_or(text, word1, word2):
    pattern = r"""
        \b{word1}\b                                 # Match word1
        \s*(?:,?\s*or\s*,?\s*)                      # Match 'or', ', or', 'or,'
        (?P<list>                                   # Start drug list
            DRUG\d+                                 # First drug
            (?:\s*,\s*DRUG\d+)*                     # Additional optional comma-separated drugs
        )
    """.format(word1=re.escape(word1))

    match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)

    if match:
        drug_list = re.findall(r'DRUG\d+', match.group('list'))
        return word2 in drug_list
    return False

def negative_filtering(pairs):
    filtered_pairs = []

    for pair in pairs:
        sentence = pair.sentence
        entity1_match = re.search(r'\[\s*e1\s*\]\s*(DRUG\d+)\s*\[\s*/\s*e1\s*\]', sentence)
        entity2_match = re.search(r'\[\s*e2\s*\]\s*(DRUG\d+)\s*\[\s*/\s*e2\s*\]', sentence)
        if not entity1_match or not entity2_match:
            continue  # skip malformed pairs

        entity1 = entity1_match.group(1)
        entity2 = entity2_match.group(1)

        is_in_par = (
            is_followed_by_examples_containing(sentence, entity1, entity2) or
            are_words_in_same_parentheses(sentence, entity1, entity2) or
            is_word_in_following_brackets(sentence, entity1, entity2) or
            is_word_followed_by_list(sentence, entity1, entity2) or
            is_word_followed_by_comma_and_or_plus(sentence, entity1, entity2) or
            is_words_between_or(sentence, entity1, entity2)
        )

        if not is_in_par:
            filtered_pairs.append(pair)

    return filtered_pairs


def is_in_par(sentence,entity1,entity2):
      return  (is_followed_by_examples_containing(sentence, entity1, entity2) or
            are_words_in_same_parentheses(sentence, entity1, entity2) or
            is_word_in_following_brackets(sentence, entity1, entity2) or
            is_word_followed_by_list(sentence, entity1, entity2) or
            is_word_followed_by_comma_and_or_plus(sentence, entity1, entity2) or
            is_words_between_or(sentence, entity1, entity2))