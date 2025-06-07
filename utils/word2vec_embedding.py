import torch
unk_cache = {}

def getting_word2vec_embedding_for_token(token,word2vec_model,word2vec_size,device):
    if token in word2vec_model.wv:
        vector = torch.tensor(word2vec_model.wv[token], dtype=torch.float32)
    else:
        if token not in unk_cache:
            unk_cache[token] = torch.randn(word2vec_size, dtype=torch.float32)
        vector = unk_cache[token]

    if torch.cuda.is_available():
        vector = vector.to(device)

    return vector