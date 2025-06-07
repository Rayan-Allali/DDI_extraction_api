import spacy
import re
split_chars= [
        "'", "\"", ".", ",", "!", "?", ":", ";",
        "(", ")", "[", "]", "{", "}", "<", ">",
        "+", "-", "=", "/", "\\", "|", "_", "~", "@", "#", "$", "%", "&", "*",
        "..." ]

nlp = spacy.load("en_core_web_sm")
def tokenize_with_custom_splits(text):
    def split_special_chars(text):
        pattern = f"({'|'.join(map(re.escape, split_chars))})"
        return [
            token
            for word in text.split()
            for token in re.split(pattern, word)
            if token
        ]

    preprocessed = " ".join(split_special_chars(text))
    return preprocessed