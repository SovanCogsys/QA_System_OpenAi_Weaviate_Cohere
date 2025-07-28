# %%
import weaviate
import os
import openai
import re
import spacy
WEAVIATE_URL = "https://yf1.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "bENJZ3l1"
OPENAI_API_KEY ="sk-proj"
os.environ["WEAVIATE_URL"] = WEAVIATE_URL
os.environ["WEAVIATE_API_KEY"] = WEAVIATE_API_KEY
os.environ["OPENAI_APIKEY"] = OPENAI_API_KEY
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.retrievers import BM25Retriever
import fitz
from langchain.schema import Document
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.load import dumps, loads
from langchain.chains import LLMChain
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser  
import base64
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

# %%
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
client = weaviate.Client(
    url=weaviate_url, auth_client_secret=weaviate.AuthApiKey(weaviate_api_key),
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")
    }
)

# %%
class_obj = {
    "class": "Electromagnetic_Waves",
    "properties": [
        {
            "name": "content",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "tokenization": "lowercase"
                }
            }
        },
        {
            "name": "chapter",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizePropertyName": False,
                    "skip": True
                }
            }
        },
        {
            "name": "title",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizePropertyName": False,
                    "skip": True
                }
            }
        },
        {
            "name": "section",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizePropertyName": False,
                    "skip": True
                }
            }
        },
        {
            "name": "source_page",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizePropertyName": False,
                    "skip": True
                }
            }
        },
        {
            "name": "doc_page",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizePropertyName": False,
                    "skip": True
                }
            }
        }
    ],
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-large"
        },
        "generative-openai": {
            "model": "gpt-4o"
        }
    }
}

client.schema.create_class(class_obj)

# %%
import pickle

chunk_file = "/home/sovan/harsh/class12_phy_ch8/phy_chap8.pkl"

# Load the pickle and print the length of the list
with open(chunk_file, "rb") as f:
    all_chunks = pickle.load(f)


print(len(all_chunks))

# %%
data_objects = []

for chunk in all_chunks:
    data_object = {
        "content": chunk["content"],
        "chapter": chunk.get("chapter", "unknown"),
        "title": chunk.get("title", "unknown"),
        "section": chunk.get("section", "unknown"),
        "source_page": chunk.get("source_page", "unknown"),
        "doc_page": chunk.get("doc_page", "unknown")
    }
    data_objects.append(data_object)

# %%
print(len(data_objects))

# %%

client.batch.configure(batch_size=10)  # Set the fixed batch size

with client.batch as batch:
    for data_object in data_objects:
        batch.add_data_object(
            data_object, 
            class_name="Electromagnetic_Waves"  # Replace with your class name
        )


# %%



