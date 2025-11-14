import os 

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

key_path = "./PRIVATE/key.txt"
project_path = "./PRIVATE/project.txt"

if "IBM_API_KEY" not in os.environ:
    with open(key_path, "r") as f:
        os.environ["IBM_API_KEY"] = f.read()

if "IBM_PROJECT_ID" not in os.environ:
    with open(project_path, "r") as f:
        os.environ["IBM_PROJECT_ID"] = f.read()

def get_embeddings():
    
    api_key = os.getenv("IBM_API_KEY")
    url = "https://us-south.ml.cloud.ibm.com"
    project_id = os.getenv("IBM_PROJECT_ID")
    model_id = "ibm/granite-embedding-278m-multilingual"

    if not api_key:
        print("IBM_API_KEY not set")
    if not project_id:
        print("IBM_PROJECT_ID not set")
    if not model_id:
        print("IBM_EMBED_MODEL not set")
    
    
    return WatsonxEmbeddings(
        model_id=model_id,
        project_id=project_id,
        url=url,
        apikey=api_key,
    )

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
