from model_registery import ModelRegistry,get_model_registry
from ddi_extraction_pipeline import ddis_extraction
from dto.text_input import TextInput
from fastapi import APIRouter,Depends
from preprocess.sentece_preprocess import blind_drug_mentions
from utils.ner_utils import predict_ner_drug
router = APIRouter()

@router.post("/extract-ddis")
async def extract_ddis_from_input(input: TextInput,models: ModelRegistry = Depends(get_model_registry)):
    print("text :",input)
    
    
    results = ddis_extraction(input.text, models)
    return results

@router.post("/extract-drugs")
async def extract_ddis_from_input(input: TextInput,models: ModelRegistry = Depends(get_model_registry)):
    print("text :",input)
      
    ner_bio_tags,sentence = predict_ner_drug(sentence,models.ner_model,models.ner_tokenizer,models.device)
    result =blind_drug_mentions(sentence,ner_bio_tags)
    
    if result is None:
        return result
    sentence, drug_map = result
    return {
            "sentence":sentence,
            "drugs_dict":drug_map
            }

