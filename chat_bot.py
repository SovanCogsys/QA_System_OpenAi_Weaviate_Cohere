# %%
from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig,pipeline,AutoModelForCausalLM
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
# import chromadb
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle
import gc

from langchain.llms import HuggingFacePipeline
from langchain.retrievers.multi_query import MultiQueryRetriever
import torch
import os
model_id_1 = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "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"


#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

# %%
class QwenEmbeddingWrapper(Embeddings):
    def __init__(self, model_id: str, task: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        cache_dir = "/scratch/work/sovan/huggingface"
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
        )
        # Quantization config
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,         # Or use load_in_8bit=True for 8-bit
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left",cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained( 
            model_id, 
            trust_remote_code=True, 
            cache_dir=cache_dir,
         quantization_config=bnb_config,
        device_map="auto")
        
        self.model.eval()
        self.task = task
        self.device = device

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _embed(self, texts , batch_size=9):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeddings.extend(F.normalize(embeddings, p=2, dim=1).cpu().tolist())
        return all_embeddings

    def embed_documents(self, texts):
        # No need to prepend task to corpus/doc chunks
        return self._embed(texts)

    def embed_query(self, text):
        # Add task instruction for query embedding
        formatted = f"Instruct: {self.task}\nQuery:{text}"
        return self._embed([formatted])[0]


# %%

def extract_columns(pdf_path):
    import fitz  # make sure this is included
    doc = fitz.open(pdf_path)
    pages = []
    page_counter = 1  # start from 1

    for _, page in enumerate(doc):
        width = page.rect.width
        height = page.rect.height
        mid_x = width / 2

        left_rect = fitz.Rect(0, 0, mid_x, height)
        right_rect = fitz.Rect(mid_x, 0, width, height)

        left_text = page.get_textbox(left_rect).strip()
        right_text = page.get_textbox(right_rect).strip()

        if left_text:
            pages.append({
                "page_content": left_text,
                "metadata": {"source": pdf_path, "pdf_page": page_counter, "column": "left"}
            })
            page_counter += 1

        if right_text:
            pages.append({
                "page_content": right_text,
                "metadata": {"source": pdf_path, "pdf_page": page_counter, "column": "right"}
            })
            page_counter += 1

    return pages

# %%
# raw_pages = extract_columns("as.pdf")
# documents = [Document(page_content=p["page_content"].lower(), metadata=p["metadata"]) for p in raw_pages]
# #Split and chunk 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
# chunks = text_splitter.split_documents(documents)



# with open("chunks_1.pkl", "wb") as f:
#     pickle.dump(chunks, f)


# %%
# for i, chunk in enumerate(chunks[:5]):  # view first 5 chunks
#     print(f"Chunk {i}")
#     print("Content:", chunk.page_content[:100], "...")  # show first 100 chars
#     print("Metadata:", chunk.metadata)
#     print("Page number:", chunk.metadata.get("pdf_page"))
#     print()


# %%
# texts = [doc.page_content for doc in chunks]
# metadatas = [doc.metadata for doc in chunks]
task = "Given a web search query, retrieve relevant passages that answer the query"

embedding = QwenEmbeddingWrapper(model_id="Qwen/Qwen3-Embedding-8B", task=task)
# print(type(embedding))
# print(type(texts))
# print(type(metadatas))
# print(texts[0])

# %%
# # 6. Create a Chroma DB using precomputed embeddings
# db = Chroma.from_documents(
#     # texts=texts,
#     documents=chunks,
#     embedding=embedding,
#     # collection_metadata=metadatas,
#     collection_name="collection_e1",
#     persist_directory="/scratch/work/sovan/chroma_db"
# )



# %%

db = Chroma(
    embedding_function=embedding,
    collection_name="collection_e1",
    persist_directory="/scratch/work/sovan/chroma_db_1"
)

# %%
#query="what is a State?"

# retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#retrieved_docs = retriever.get_relevant_documents(query)


# %%

with open("chunks_1.pkl", "rb") as f:
    chunks = pickle.load(f)

retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 12})
# Create BM25Retriever from the documents
bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=12)

# %%


ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.65, 0.35]
)


# question ="what is a waldsterben ? "
# question ="what is a superquadra ?"
# question ="What is a state ?"
# question = "What  did benjamin constant say ?"
question ="Tell something about politics of measurement?"
#question ="Tell something about statecraft and the hieroglyphics of measurement?"

retrieved_docs = ensemble_retriever.invoke(question)

# %%
# Step 1: Extract page numbers
page_numbers = set()  # use a set to avoid duplicates
for doc in retrieved_docs:
    page = doc.metadata.get("pdf_page")
    if page is not None:
        page_numbers.add(page)

# Convert to sorted list (optional)
page_numbers = sorted(list(page_numbers))

# Step 2: Store in pickle
with open("retrieved_pages.pkl", "wb") as f:
    pickle.dump(page_numbers, f)

print("Stored page numbers:", page_numbers)

# %%
#print(len(retrieved_docs))
#print(retrieved_docs)
unique_docs = []
seen_content = set()
for doc in retrieved_docs:
    if doc.page_content not in seen_content:
        unique_docs.append(doc)
        seen_content.add(doc.page_content)

# Combine retrieved docs into a context string
context = "\n\n".join([doc.page_content for doc in unique_docs])
#print(context)

# %%
# Save

with open("context.pkl", "wb") as f:
    pickle.dump(context, f)


with open("question.pkl", "wb") as f:
    pickle.dump(question, f)


# Step 1: Delete model and tokenizer
del embedding.model
del embedding.tokenizer
del embedding  

# Step 2: Garbage collection
gc.collect()

# Step 3: Clear GPU cache
torch.cuda.empty_cache()

# %%
# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,         # Or use load_in_8bit=True for 8-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

cache_dir = "/scratch/work/sovan/huggingface"

tokenizer = AutoTokenizer.from_pretrained(model_id_1, token=hf_token, cache_dir= cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id_1, token=hf_token,cache_dir= cache_dir
                                             ,quantization_config=bnb_config
                                            )

# %%
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",  # 0 means first GPU
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=False
)

# %%
# Load context and query from pickled files
with open("context.pkl", "rb") as f:
    context = pickle.load(f)

with open("question.pkl", "rb") as f:
    query = pickle.load(f)

# Format prompt using LLaMA 3 instruct style
prompt = f"""<|begin_of_text|><|system|>
You are a helpful assistant. Use only the provided context to answer the question. 
Answer naturally, as if you're responding directly to the question.
<|end|><|user|>
[CONTEXT]
{context}

[QUESTION]
{query}

[ANSWER]
<|end|><|assistant|>
"""

# %%
# Generate answer
response = generator(prompt)[0]["generated_text"]

# Extract and print the assistant's answer
# (If LLaMA 3 repeats prompt, split to get final answer)
if "[ANSWER]" in response:
    answer = response.split("[ANSWER]")[-1].strip()
else:
    answer = response.strip()

print("\n=== Model Answer ===\n")
print(answer)


# %%
# Step 1: Delete the pipeline, model, tokenizer
del generator
del model
del tokenizer

# Step 2: Collect garbage
gc.collect()

# Step 3: Clear CUDA cache
torch.cuda.empty_cache()


