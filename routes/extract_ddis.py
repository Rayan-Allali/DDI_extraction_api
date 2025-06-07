from model_registery import ModelRegistry,get_model_registry
from ddi_extraction_pipeline import ddis_extraction
from dto.text_input import TextInput
from fastapi import APIRouter,Depends

router = APIRouter()

@router.post("/extract-ddis")
async def extract_ddis_from_input(input: TextInput,models: ModelRegistry = Depends(get_model_registry)):
    print("text :",input)
    
    
    results = ddis_extraction(input.text, models)
    return results