
import os
import weaviate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.output_parsers import StrOutputParser
WEAVIATE_URL = "https://yf1xhun0ttywarwcufiqow.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "bENJZ3l1bTExNWZTcGhaY19NT0l0a0g3QnA0UnM1NGFON2w1UUwzcWZRVjIySHdDZm53VnBveFhwN0I0PV92MjAw"
OPENAI_API_KEY = "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"
COHERE_API_KEY = "AQrl2dSsLLH2lk41BrscjvNyllaqlMwd0jt9z50i"

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

compressor = CohereRerank(
    cohere_api_key=COHERE_API_KEY,
    model="rerank-english-v3.0",
    top_n=6
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable assistant. Use the context below, which contains excerpts from different documents along with their sources and relevance scores, to answer the question naturally and confidently.

- Do NOT mention "document", "source", or "relevance score" in your answer.
- Synthesize the answer as if you already knew the information.
- Focus on accuracy, clarity, and completeness.
- If multiple documents mention relevant facts, combine them intelligently.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ==================== MAIN CLASS ====================

class SubjectQA:
    def __init__(self):
        self.client = client
        self.llm = llm
        self.compressor = compressor
        self.custom_prompt = custom_prompt
        self.parser = StrOutputParser()

    def get_retriever(self, collection_name):
        base_retriever = WeaviateHybridSearchRetriever(
            client=self.client,
            index_name=collection_name,
            text_key="content",
            attributes=["chapter", "source_page", "doc_page", "section", "title"],
            alpha=0.6,
            k=15,
            create_schema_if_missing=True
        )

        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )

    def extract_context(self, query, user_subject):
        collection_map = {
            "physics": "Electricity",
            "biology": "The_Living_World",
            "chemistry": "Redox_Reactions",
        }

        if user_subject not in collection_map:
            raise ValueError("Invalid subject. Choose from Physics, Chemistry, or Biology.")

        collection_name = collection_map[user_subject]
        retriever = self.get_retriever(collection_name)

        docs = retriever.invoke(query)

        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("relevance_score", 0) > 0.00
        ]

        seen_contents = set()
        unique_filtered_docs = []
        for doc in filtered_docs:
            content = doc.page_content.strip()
            if content not in seen_contents:
                seen_contents.add(content)
                unique_filtered_docs.append(doc)

        context = "\n\n".join(
            f"### Document {i + 1}\n"
            f"Source: {doc.metadata.get('chapter', 'N/A')}, Relevance: {doc.metadata.get('relevance_score', 0):.2f}\n"
            f"{doc.page_content.strip()}"
            for i, doc in enumerate(unique_filtered_docs)
        )

        return {
            "context": context,
            "question": query,
            "filtered_docs": unique_filtered_docs
        }

    def get_answer(self, user_subject, query, user_response):
        context_output = self.extract_context(query, user_subject)

        final_input = {
            "context": context_output["context"],
            "question": context_output["question"]
        }

        response = (self.custom_prompt | self.llm | self.parser).invoke(final_input)

        return response, context_output["filtered_docs"]

# ==================== USER_Interaction====================


qa_system = SubjectQA()

user_subject = input("Enter the Document name (Physics or Chemistry or Biology): ").strip().lower()
query = input("Enter your Question: ")
user_response = input("Enter your answer: ")

response, filtered_docs = qa_system.get_answer(user_subject, query, user_response)

print("\n✅ Answer:\n")
print(response)

print("\n📚 Sources Referenced:\n")
print("{:<6} {:<30} {}".format("Page", "Source", "Snippet"))
for doc in filtered_docs:
    page = doc.metadata.get('doc_page', 'N/A')
    source = doc.metadata.get('chapter', 'Unknown')
    snippet = doc.page_content.strip().replace("\n", " ")
    print("{:<6} {:} {}".format(page, source, snippet + "..."))
