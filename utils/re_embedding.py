import torch
import re
def get_cls_embedding_from_last_four_layers(last_four_layers):
    cls_embeddings = last_four_layers[:, 0, 0, :]
    cls_embedding_avg = cls_embeddings.mean(dim=0)
    return cls_embedding_avg

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[e1]", " " "[/e1]", "[e2]", "[/e2]"}

def get_biobert_embeddings_last_four_layers_for_each_token(text,re_model,re_tokenizer,device):
    inputs = re_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    tokens = re_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = re_model(**inputs, output_hidden_states=True)
    last_4_layers = torch.stack(outputs.hidden_states[-4:])
    token_embeddings = last_4_layers.mean(dim=0)[0]

    for i, token in enumerate(tokens) :
      if token not in SPECIAL_TOKENS :
        print('tokens' , token)
    filtered_embeddings = [
        token_embeddings[i] for i, token in enumerate(tokens) if token not in SPECIAL_TOKENS
    ]

    if not filtered_embeddings:
        return torch.zeros(re_model.config.hidden_size).to(device)
    return filtered_embeddings


from collections import defaultdict
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[e1]", "[/e1]", "[e2]", "[/e2]" , " "}

def get_entity_marker_word_ids(tokens, word_ids):
    e1_word_id = e2_word_id = e1_end_word_id = e2_end_word_id = None

    for token, word_id in zip(tokens, word_ids):
        if token == '[e1]':
            e1_word_id = word_id
        elif token == '[e2]':
            e2_word_id = word_id
        elif token == '[/e1]':
            e1_end_word_id = word_id
        elif token == '[/e2]':
            e2_end_word_id = word_id

    return e1_word_id, e1_end_word_id, e2_word_id, e2_end_word_id

def get_cleaned_words_using_biobert_tokenizer(text,re_tokenizer,device):
    inputs = re_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    tokens = re_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids()

    word_id_to_tokens = defaultdict(list)

    e1_word_id, e1_end_word_id, e2_word_id, e2_end_word_id = get_entity_marker_word_ids(tokens, word_ids)

    for token, word_id in zip(tokens, word_ids):
        if word_id is None or token in SPECIAL_TOKENS:
            continue
        word_id_to_tokens[word_id].append(token)

    words = []
    for word_id in sorted(word_id_to_tokens.keys()):
        sub_tokens = word_id_to_tokens[word_id]
        word = "".join([t.replace("##", "") if t.startswith("##") else t for t in sub_tokens])
        if e1_word_id < word_id < e1_end_word_id:
            word = "DRUG_E1"
        elif e2_word_id < word_id < e2_end_word_id:
            word = "DRUG_E2"
        word = re.sub(r'drug\d+', 'DRUG_E', word, flags=re.IGNORECASE)
        words.append(word)

    return words