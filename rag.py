import os 

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxEmbeddings

key_path = "./PRIVATE/key.txt"
project_path = "./PRIVATE/project.txt"

def get_embeddings():
    api_key = os.getenv("IBM_API_KEY")
    url = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("IBM_PROJECT_ID")
    model_id = os.getenv("IBM_EMBED_MODEL")

    if not api_key:
        print("IBM_API_KEY not set")
    if not project_id:
        print("IBM_PROJECT_ID not set")
    if not model_id:
        print("IBM_EMBED_MODEL not set")
    
    creds = Credentials(url=url, api_key=api_key)
    return WatsonxEmbeddings(credentials=creds, project_id=project_id, model_id=model_id)

def load_documents(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store(paths):
    all_chunks = []
    for p in paths:
        all_chunks.extend(load_documents(p))
    if not all_chunks:
        return None

    embeddings = get_embeddings()
    vs = FAISS.from_documents(all_chunks, embeddings)
    return vs

def retrieve_once(paths, query, k=4):
    vs = create_vector_store(paths)
    if not vs:
        return []
    return vs.similarity_search(query, k=k)
