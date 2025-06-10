from preprocess.sentece_preprocess import blind_drug_mentions
from utils.gat.createGraphDataset import create_graph_dataset
from utils.ner_utils import predict_ner_drug
from generate_drug_pairs import generate_drugPairSentences_for_pairs
from torch_geometric.data import Data, DataLoader as DataLoadergGometric
from utils.gat.gat_utils import re_prediction_with_gat
from model_registery import ModelRegistry
def ddis_extraction(sentence, models: ModelRegistry):
  ner_bio_tags,sentence = predict_ner_drug(sentence,models.ner_model,models.ner_tokenizer,models.device)
  result =blind_drug_mentions(sentence,ner_bio_tags)
  
  if result is None:
    return result
  sentence, drug_map = result
  print(sentence)
  generated_DrugPairSentences = generate_drugPairSentences_for_pairs(sentence, drug_map)
  print(len(generated_DrugPairSentences))

  if len(generated_DrugPairSentences) == 0 :
    return None
  data_graph=create_graph_dataset(generated_DrugPairSentences,models.re_biobert_model,models.re_biobert_tokenizer,models.word2vec_model,models.word2vec_model_size, models.spacy_nlp ,models.stanza_nlp,models.device)
  data_loader_graph = DataLoadergGometric(data_graph, batch_size=128,shuffle=True)
  results =re_prediction_with_gat(data_loader_graph,models.re_gat_model,models.device)
  return results
