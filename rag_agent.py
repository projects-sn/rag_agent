from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate

import pandas as pd
import streamlit as st
import re

OPENAI_API_KEY = "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"

df = pd.read_csv("meeting_summaries.csv").dropna(subset=["Сводка"])

df["metadata"] = df.apply(lambda row: {
    "doc_id": str(row["Документ"]),
    "date": str(row["Дата"])
}, axis=1)

df["text"] = (
    "Дата: " + df["Дата"] +
    "\nДокумент №: " + df["Документ"].astype(str) +
    "\n\n" + df["Сводка"]
)

docs = [Document(page_content=row["text"], metadata=row["metadata"]) for _, row in df.iterrows()]

# Сплиттер токенов
splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=40)
split_docs = splitter.split_documents(docs)

# Эмбеддинги
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# Векторный поиск через DocArray
vectorstore = DocArrayInMemorySearch.from_documents(split_docs, embeddings)

# BM25
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 4

# Гибридный ретривер
def hybrid_retrieve(query: str, vectorstore, bm25_retriever, k=4):
    vector_docs = vectorstore.similarity_search(query, k=k)
    bm25_docs = bm25_retriever.get_relevant_documents(query)

    all_docs = vector_docs + bm25_docs
    unique = {}
    for doc in all_docs:
        key = doc.page_content[:200]
        if key not in unique:
            unique[key] = doc

    return list(unique.values())[:k]

# Извлечение даты/номера из запроса
def extract_date_and_doc_id(query: str):
    date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", query)
    doc_match = re.search(r"(документ|встреча)\s*№?\s*(\d+)", query.lower())
    date = date_match.group(1) if date_match else None
    doc_id = doc_match.group(2) if doc_match else None
    return date, doc_id

# Кастомный ретривер с фильтрацией
class CustomRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        date, doc_id = extract_date_and_doc_id(query)

        filtered_docs = split_docs
        if date or doc_id:
            filtered_docs = [
                doc for doc in split_docs
                if (not date or doc.metadata.get("date") == date)
                and (not doc_id or doc.metadata.get("doc_id") == doc_id)
            ]
            if not filtered_docs:
                print("⚠️ Нет совпадений по метаданным, используем весь корпус.")
                filtered_docs = split_docs

            temp_vectorstore = DocArrayInMemorySearch.from_documents(filtered_docs, embeddings)
            temp_bm25 = BM25Retriever.from_documents(filtered_docs)
            temp_bm25.k = 4

            return hybrid_retrieve(query, temp_vectorstore, temp_bm25, k=4)

        return hybrid_retrieve(query, vectorstore, bm25_retriever, k=4)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
                Ты — помощник, анализирующий стенограммы встреч.
                Ответь **полно и развёрнуто**, используя релевантную информацию из встреч, переданную в контексте.
                Отвечай детально и развернуто, упоминай источники и ответственных людей.

                Контекст:
                {context}

                Вопрос:
                {question}

                Ответ:
                """
)

def build_rag_chain():
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2,
        model_name="gpt-4",
        max_tokens=2048
    )

    retriever = CustomRetriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
