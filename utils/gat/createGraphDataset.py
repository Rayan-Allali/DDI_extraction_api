import torch
from torch_geometric.data import Data
from utils.nlp.stanza_utils import extract_dependency_tree_stanford
from utils.re_word2vec_embedding import get_word_embeddings_and_tokens_biobert_word2vec
def sentence_to_graph_stanza_bert_token_word2vec(sentence_from_biobert,x_embeddings,nlp,stanza_nlp):
    doc = nlp(sentence_from_biobert)
    tokens, edges=extract_dependency_tree_stanford(doc,stanza_nlp)
    edge_index = torch.tensor(
            [(child, parent) for child, parent, _ in edges],
            dtype=torch.long
        ).t().contiguous()
    return Data(x=x_embeddings, edge_index=edge_index, num_nodes=len(tokens)), tokens


def create_graph_dataset(pairs,re_model,re_tokenizer,word2vec_model,word2vec_size,nlp,stanza_nlp,device):
    data_list = []
    i=0
    for pair in pairs:
       sentence = pair.sentence
       embeddings,words,cls_embedding=get_word_embeddings_and_tokens_biobert_word2vec(sentence,re_model,re_tokenizer,word2vec_model,word2vec_size,device)
       sentence_from_biobert=' '.join(words)
       graph_data, tokens = sentence_to_graph_stanza_bert_token_word2vec(sentence_from_biobert,embeddings,nlp,stanza_nlp)
       graph_data.biobert_cls = cls_embedding
       graph_data.sentence = sentence
       graph_data.drug1 = pair.drug1
       graph_data.drug2 = pair.drug2
       data_list.append(graph_data)
       if graph_data.x.shape[0] != graph_data.num_nodes :
           print('i ',i)
           print('sentence ',sentence)
           print('sentence_from_biobert ',sentence_from_biobert)
           print('graph_data ',graph_data)
           break
       i=i+1
    return data_list