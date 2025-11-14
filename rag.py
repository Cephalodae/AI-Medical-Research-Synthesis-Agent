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

os.environ.setdefault("IBM_URL", "https://us-south.ml.cloud.ibm.com")
os.environ.setdefault("IBM_EMBED_MODEL", "ibm/granite-embedding-107m-multilingual")

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

def get_llm():
    model_id = "ibm/granite-13b-chat-v2" 
    api_key    = os.environ["IBM_API_KEY"]
    project_id = os.environ["IBM_PROJECT_ID"]
    creds = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)

    return WatsonxLLM(credentials=creds, project_id=project_id, model_id=model_id, params={
            "max_new_tokens": 400,
            "temperature": 0.2,
            "top_p": 0.9,
        },)    


system_prompt = """You are a careful research assistant.
Answer ONLY using the supplied context between <<<CONTEXT>>> and <<<END>>>.
If the answer is not present, say "I don't know."
Be concise and factual."""

template = PromptTemplate.from_template(
    system_prompt + "\n\n<<<CONTEXT>>>\n{context}\n<<<END>>>\n\nQuestion: {question}\nAnswer:"
)

def get_answer(paths, query):
    vs = create_vector_store(paths)
    if not vs:
        return "No documents provided."

    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": template},
        return_source_documents=True,
    )

    result = qa_chain({"query": query})
    return result
