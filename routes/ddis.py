from model_registery import ModelRegistry,get_model_registry
from ddi_extraction_pipeline import ddis_extraction
from dto.text_input import TextInput
from fastapi import APIRouter,Depends
from preprocess.sentece_preprocess import extracting_drugs
from utils.ner_utils import predict_ner_drug
import torch
router = APIRouter()

@router.post("/extract-ddis")
async def extract_ddis_from_input(input: TextInput,models: ModelRegistry = Depends(get_model_registry)):
    torch.cuda.empty_cache()
    results = ddis_extraction(input.text, models)
    return results

@router.post("/extract-drugs")
async def extract_drugs_from_input(input: TextInput,models: ModelRegistry = Depends(get_model_registry)):
    ner_bio_tags,sentence = predict_ner_drug(sentence=input.text,ner_model=models.ner_model,ner_tokenizer=models.ner_tokenizer,device=models.device)
    result =extracting_drugs(sentence,ner_bio_tags)
    return result

