from preprocess.sentece_preprocess import blind_drug_mentions
from utils.gat.createGraphDataset import create_graph_dataset
from utils.ner_utils import predict_ner_drug
from generate_drug_pairs import generate_drugPairSentences_for_pairs
from torch_geometric.data import Data, DataLoader as DataLoadergGometric
from utils.gat.gat_utils import re_prediction_with_ensemble_gat
from model_registery import ModelRegistry
import torch
import gc
def ddis_extraction(sentence, models: ModelRegistry):
  ner_bio_tags = None
  result = None
  results = None
  filtered_results = None
  data_graph = None
  data_loader_graph = None
  gat_results = None
  try:
    ner_bio_tags,sentence = predict_ner_drug(sentence,models.ner_model,models.ner_tokenizer,models.device)
    result =blind_drug_mentions(sentence,ner_bio_tags)
    
    if result is None:
      return result
    sentence, drug_map = result
    results,filtered_results= generate_drugPairSentences_for_pairs(sentence, drug_map)
    print(len(results))
    print(len(filtered_results))
    if len(results) == 0 :
      return None
    data_graph=create_graph_dataset(results,models.re_biobert_model,models.re_biobert_tokenizer,models.word2vec_model,models.word2vec_model_size, models.spacy_nlp ,models.stanza_nlp,models.device)
    data_loader_graph = DataLoadergGometric(data_graph, batch_size=128,shuffle=True)
    results =re_prediction_with_ensemble_gat(data_loader_graph,models.re_gat_models,models.device)
    print(len(results + filtered_results))
    return results + filtered_results  
  finally:
        del ner_bio_tags, result, results, filtered_results, data_graph, data_loader_graph, gat_results
        gc.collect()
        torch.cuda.empty_cache()
