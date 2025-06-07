import torch
import re
from utils.re_embedding import get_cls_embedding_from_last_four_layers, get_entity_marker_word_ids,SPECIAL_TOKENS
from utils.word2vec_embedding import getting_word2vec_embedding_for_token
from collections import defaultdict

def combine_word2vec_biobert_embed(biobert_embed, word2vec_embed):
    return torch.cat((biobert_embed, word2vec_embed), dim=0)



# re_model,re_tokenizer,gat_model,word2vec_model
def get_word_embeddings_and_tokens_biobert_word2vec(text,re_model,re_tokenizer,word2vec_model,word2vec_size,device):
    inputs = re_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    tokens = re_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids()
    words =[]
    with torch.no_grad():
        outputs = re_model(**inputs, output_hidden_states=True)

    last_4_layers = torch.stack(outputs.hidden_states[-4:])

    cls_embedding=get_cls_embedding_from_last_four_layers(last_4_layers)

    token_embeddings = last_4_layers.mean(dim=0)[0]
    word_id_to_embeddings = defaultdict(list)
    word_id_to_tokens = defaultdict(list)

    e1_word_id, e1_end_word_id, e2_word_id, e2_end_word_id = get_entity_marker_word_ids(tokens, word_ids)
    for i, (token, word_id) in enumerate(zip(tokens, word_ids)):
        if word_id is None or token in SPECIAL_TOKENS:
            continue
        word_id_to_embeddings[word_id].append(token_embeddings[i])
        word_id_to_tokens[word_id].append(token)
    word_embeddings = []
    words = []

    for word_id in sorted(word_id_to_embeddings.keys()):
        sub_tokens = word_id_to_tokens[word_id]
        word = "".join([t.replace("##", "") if t.startswith("##") else t for t in sub_tokens])
        if word_id > e1_word_id and word_id < e1_end_word_id:
            word = "DRUG_E1"
        elif word_id > e2_word_id and word_id < e2_end_word_id:
            word = "DRUG_E2"
        word = re.sub(r'drug\d+', 'DRUG_E', word, flags=re.IGNORECASE)
        words.append(word)
        word2vec_embed =getting_word2vec_embedding_for_token(word,word2vec_model,word2vec_size,device)
        subtoken_embeds = word_id_to_embeddings[word_id]
        avg_embed = torch.stack(subtoken_embeds).mean(dim=0)
        biobert_word_2_vec_embedding = combine_word2vec_biobert_embed(avg_embed,word2vec_embed)
        word_embeddings.append(biobert_word_2_vec_embedding)

    if not word_embeddings:
        return torch.zeros(re_model.config.hidden_size).to(device), []

    return torch.stack(word_embeddings),words,cls_embedding

