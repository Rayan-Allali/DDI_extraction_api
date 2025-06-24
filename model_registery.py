from transformers import  AutoTokenizer
import torch

from fastapi import Request
from utils.gat.gat_model import GATModelWithAttention
from pathlib import Path
from transformers import BertForTokenClassification, BertTokenizerFast, AutoModelForSequenceClassification, AutoTokenizer
from gensim.models import Word2Vec
import spacy
import stanza
from pathlib import Path
import torch
models_path_and_args = [
    {
        "path": "best_model_f1_0.8105_bert_word2vec.pt",
        "node_in_dim": 1068,
        "dropout_rate": 0.3485807291196686,
    },
    {
        "path": "best_model_f1_0.8109_bert_word2vec.pt",
        "node_in_dim": 1068,
        "dropout_rate": 0.3485807291196686,
    },
    {
        "path": "best_model_f1_0.8114_bert_word2vec.pt",
        "node_in_dim": 1068,
        "dropout_rate": 0.3485807291196686,
    },
]

class ModelRegistry:
    def __init__(self):
        try:
            project_root = Path(__file__).parent
           
            models_dir = project_root / "models"
            print("models_dir ", models_dir)
            gat_models_dir = models_dir / "gat_models"
            if not models_dir.exists():
                raise FileNotFoundError(f"Models directory not found at: {models_dir}")
            if not gat_models_dir.exists():
                raise FileNotFoundError(f"GAT models directory not found at: {gat_models_dir}")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("self.device  ", self.device )
            
            re_biobert_model_path = models_dir / "re_biobert_model"
            if not re_biobert_model_path.exists():
                raise FileNotFoundError(f"RE BioBERT model path not found: {re_biobert_model_path}")
            self.re_biobert_tokenizer = AutoTokenizer.from_pretrained(re_biobert_model_path)
            self.re_biobert_model = AutoModelForSequenceClassification.from_pretrained(re_biobert_model_path, device_map=None).to(self.device)
            self.re_biobert_model.eval()
            ner_model_path = models_dir / "ner_biobert_model"
            if not ner_model_path.exists():
                raise FileNotFoundError(f"NER model path not found: {ner_model_path}")
            self.ner_tokenizer = BertTokenizerFast.from_pretrained(ner_model_path)
            self.ner_model = BertForTokenClassification.from_pretrained(ner_model_path, device_map=None).to(self.device)
            self.ner_model.eval()

            gat_model_path = gat_models_dir / models_path_and_args[2]["path"]
            if not gat_model_path.exists():
                raise FileNotFoundError(f"GAT model weights not found: {gat_model_path}")
            self.re_gat_model = GATModelWithAttention(
                node_in_dim=models_path_and_args[2]["node_in_dim"],
                gat_hidden_channels=256,
                cls_dim=768,
                num_classes=5,
                dropout_rate=models_path_and_args[2]["dropout_rate"]
            ).to(self.device)
            state_dict = torch.load(gat_model_path, map_location=self.device)
            self.re_gat_model.load_state_dict(state_dict)
            self.re_gat_model.eval()

            word2vec_model_path = models_dir / "ddi_word2vec_unaugmented.model"
            if not word2vec_model_path.exists():
                raise FileNotFoundError(f"Word2Vec model not found: {word2vec_model_path}")
            self.word2vec_model = Word2Vec.load(str(word2vec_model_path))
            self.word2vec_model_size = 300

            self.spacy_nlp = spacy.load("en_core_web_sm")
            self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry
