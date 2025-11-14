#!/usr/bin/env python
# coding: utf-8

# !pip install "langchain"
# !pip install "langchain-ibm"
# !pip install "langchain-community"
# !pip install pydantic-core
# !pip install "langchain-chroma"
# !pip install langchain-pymupdf4llm
# !pip install langchain_text_splitters

import os
from langchain_ibm import WatsonxEmbeddings, ChatWatsonx
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from getpass import getpass
from ibm_watsonx_ai import Credentials
from langchain_core.prompts import ChatPromptTemplate

# config and setting up local keys and such
# -- MOST IMPORTANT (run once when imported) --
key_path = "./PRIVATE/key.txt"
key = ""
project_path = "./PRIVATE/pID.txt"
project = ""

try:
    with open(key_path, 'r') as file:
        key = file.read().strip()
except FileNotFoundError:
        print(f"Error: The file '{key_path}' was not found.")
except Exception as e:
    print(f"An error occured: {e}")
try:
    with open(project_path, 'r') as file:
        project = file.read().strip()
except FileNotFoundError:
        print(f"Error: The file '{key_path}' was not found.")
except Exception as e:
    print(f"An error occured: {e}")

# last check if env variables
if not key: key = os.environ.get("WATSONX_APIKEY")
if not project: project = os.environ.get("WATSONX_PROJECT_ID")

# just stop everything if the key or project id's not loaded here
if not key or not project:
    raise ValueError("Stopping execution: API Key or Project ID is missing/empty.")

os.environ["WATSONX_APIKEY"] = key

# -- Initialize global models ---

embeddings = WatsonxEmbeddings(
    model_id = "ibm/granite-embedding-278m-multilingual",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project,
    params={"decoding_method": "greedy"}
)

parameters = {
    "temperature": 0.6,
    "max_tokens": 1000, #since medical answers and synthesis tend to be on the longer side
}

chat = ChatWatsonx(
    model_id = "ibm/granite-3-3-8b-instruct",
    url = "https://us-south.ml.cloud.ibm.com",
    project_id = project,
    params = parameters,
)

# helper functions (we need)

# the citation formater
def format_docs_with_sources(docs):
    formatted_text = ""
    for doc in docs:
        # extract metadata
        source = doc.metadata.get("source", "Unknown File")
        page = doc.metadata.get("page", "Unkown Page")

        # modify the content to physically include source ID like [Source: medical_study.pdf, Page 2]
        content = doc.page_content.replace("\n", " ")
        formatted_text += f"Content: {content}\nSource: [{source}, Page {page}]\n\n"
    return formatted_text

def create_rag_chain(pdf_paths):
    all_docs = []

    # load the pdfs - pymupdf loader one file at a time, so we loop
    print(f"Processing {len(pdf_paths)} files..")
    for path in pdf_paths:
        loader = PyMuPDF4LLMLoader(path)
        all_docs.extend(loader.load())

    if not all_docs:
        print("No docs were loaded")
        return None

    # Split Text
    # we use a large chunk size (1000) to keep medical concepts together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = text_splitter.split_documents(all_docs)

    # Create vector store (in memory for this sesh)
    vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 4})

    # how granite would react to your messages and such
    system_instruct = """You are an expert research assistant.
    Your task is to answer the user's question by synthesizing information from provided documents.

    Strict Rules:
    1. Grounding: Base your answer exclusively on the "Provided Documents".
    2. Citation: You must cite the source using the specific filename and page number provided in the text.
       - Format: (Filename, Page X)
       - Example: (study_results.pdf, Page 5)
    3. Consolidation: If multiple sentences come from the same location, use one citation at the end of the block.
    4. Unknown: If the documents don't answer it, say "The provided documents do not contain information to answer this question."
    """

    # for the template we're gonna get the context (which would be the docs) and the {question} (the user input)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruct),
        ("human", "Provided Documents:\n{context}\n\nUser Question:\n{question}")
    ])

    # this makes it so it has the context, docs with sources
    rag_chain = (
        {"context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough()
        }
        | prompt
        | chat
    )

    return rag_chain



















# #this is a mock area for testing
# doc1 = "[DOC_1] The Granite model is developed by IBM"
# doc2 = "[DOC_2] IBM's headquarters are located in Armonk, New York"
# doc3 = "[DOC_3] Python's a programming language released in 1991"
#
# context_text = f"{doc1}\n{doc2}\n{doc3}"
#
# question_text = "who developed Granite and where are they located?" #should use doc 1 and 2 (hopefully)
# # it most def does, so we got through this part all good
#
# print(f"---INPUT---\nContext:\n{context_text}\n\nQuestion: {question_text}\n")
#
# response = rag_chain.invoke({
#     "context" : context_text,
#     "question" : question_text
# })
#
# print(f"---OUTPUT---\nb{response.content}")
#
#
