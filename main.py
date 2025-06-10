from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_registery import ModelRegistry
import torch
from routes.ddis import router as extract_ddis_router


from contextlib import asynccontextmanager

model_registry = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_registry
    from model_registery import ModelRegistry

    model_registry = ModelRegistry()
    try:
        with torch.no_grad():
            dummy_input = model_registry.ner_tokenizer("test", return_tensors="pt")
            model_registry.ner_model(**dummy_input)
            model_registry.re_biobert_model(**dummy_input)
        model_registry.spacy_nlp("test")
        model_registry.stanza_nlp("test")
        app.state.model_registry = model_registry
    except Exception as e:
        raise RuntimeError(f"Model warmup failed: {str(e)}")
    
    yield 
    if model_registry:
        del model_registry.ner_model
        del model_registry.re_biobert_model
        del model_registry.re_gat_model
        del model_registry.word2vec_model
        del model_registry.spacy_nlp
        del model_registry.stanza_nlp
        torch.cuda.empty_cache()
        model_registry = None
        app.state.model_registry = None


app = FastAPI(lifespan=lifespan)

origins = [
    "https://your-frontend-domain.com",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)
@app.get("/")
async def read_root():
    return {"message": "Welcome To DDI Extraction it is Working Fine !"}

    
app.include_router(extract_ddis_router, prefix="")