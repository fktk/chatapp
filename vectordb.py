from uuid import uuid4
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model='granite-embedding:278m')

vector_store = Chroma(
    collection_name='my_collection',
    embedding_function=embeddings,
    persist_directory='./vector_db',
)

def main():
    documents = [
        dict(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            source='tweet',
        ),
        dict(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            source="news",
        ),
        dict(
            page_content="Building an exciting new project with LangChain - come check it out!",
            source='tweet',
        ),
        dict(
            page_content="Robbers broke into the city bank and stole $1 million in cash.",
            source="news",
        ),
        dict(
            page_content="Wow! That was an amazing movie. I can't wait to see it again.",
            source='tweet',
        ),
        dict(
            page_content="Is the new iPhone worth the price? Read this review to find out.",
            source='website',
        ),
        dict(
            page_content="The top 10 soccer players in the world right now.",
            source='website',
        ),
        dict(
            page_content="LangGraph is the best framework for building stateful, agentic applications!",
            source='tweet',
        ),
        dict(
            page_content="The stock market is down 500 points today due to fears of a recession.",
            source="news",
        ),
        dict(
            page_content="I have a bad feeling I am going to get deleted :(",
            source='tweet',
        ),
    ]

    for doc in documents:
        add_items(**doc)

    query_items()


def query_items():
    results = vector_store.similarity_search(
        'LangChain provides abstractions to make working with LLMs easy',
        k=2,
    )
    for res in results:
        print(res)


def get_document_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def add_items(page_content: str, source: str):
    doc_hash = get_document_hash(page_content)
    existing_docs = vector_store.get(where={'hash': doc_hash})
    if len(existing_docs['ids']) > 0:
        print('duplicate')
        return

    vector_store.add_texts([page_content], metadatas=[{'hash': doc_hash, 'source': source}])


if __name__ == '__main__':
    main()
