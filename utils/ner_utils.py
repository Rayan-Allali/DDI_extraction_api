from utils.nlp.spacy_utils import tokenize_with_custom_splits
import torch

label_mapping = {
    "none": 0,
    "effect": 1,
    "mechanism": 2,
    "advise": 3,
    "int": 4 }
label_mapping_reverse = {
    0: "none",
    1: "effect",
    2: "mechanism",
    3: "advise",
    4: "int"
}
bio_labels = ["B-DRUG", "I-DRUG", "O"]


def extract_drug_entities_from_bio(words, ner_bio_tags):
    drugs = []
    word = ""
    for idx, label in ner_bio_tags.items():
        if label == "B-DRUG":
             if word != "":
                drugs.append(word.strip())
                word = ""
             word = words[idx]
        elif label == "I-DRUG":
            if word != "":
              word += f' {words[idx]}'
            else:
               word = words[idx] 
        else:
            if word != "":
                drugs.append(word.strip())
                word = ""

    if word:
        drugs.append(word.strip())

    return drugs


def predict_ner_drug(sentence,ner_model,ner_tokenizer,device):
    sentence = sentence.lower()
    sentence=tokenize_with_custom_splits(sentence.lower())

    tokens = ner_tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True,
                       is_split_into_words=False)

    word_ids = tokens.word_ids()
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        outputs = ner_model(**tokens)

    predicted_class_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()

    tokens_decoded = ner_tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())

    label_map = {0: "O", 1: "B-DRUG", 2: "I-DRUG", -100: "IGNORE"}
    predicted_labels = [label_map[label] for label in predicted_class_ids]
    first_labels = {}
   
    for token, label, word_id in zip(tokens_decoded, predicted_labels, word_ids):
        if word_id is not None and word_id not in first_labels:
            first_labels[word_id] = label
    return first_labels,sentence