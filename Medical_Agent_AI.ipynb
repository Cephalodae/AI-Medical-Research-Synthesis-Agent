{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "8edcf443-f7e3-445b-81fd-13c3f5afdb11",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Name: langchain-text-splitters\n",
      "Version: 1.0.0\n",
      "Summary: LangChain text splitting utilities\n",
      "Home-page: https://docs.langchain.com/\n",
      "Author: \n",
      "Author-email: \n",
      "License: MIT\n",
      "Location: /home/smiley/anaconda3/lib/python3.12/site-packages\n",
      "Requires: langchain-core\n",
      "Required-by: langchain-classic\n"
     ]
    }
   ],
   "source": [
    "# !pip install \"langchain\" \n",
    "# !pip install \"langchain-ibm\" \n",
    "# !pip install \"langchain-community\" \n",
    "# !pip install pydantic-core \n",
    "# !pip install \"langchain-chroma\"\n",
    "# !pip install langchain-pymupdf4llm\n",
    "# !pip install langchain_text_splitters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "08bab755-0055-486e-ab9a-536c38dfbe63",
   "metadata": {},
   "outputs": [],
   "source": [
    "from getpass import getpass\n",
    "from ibm_watsonx_ai import Credentials\n",
    "import os\n",
    "\n",
    "key_path = \"./PRIVATE/key.txt\"\n",
    "key = \"\"\n",
    "project_path = \"./PRIVATE/pID.txt\"\n",
    "project = \"\"\n",
    "try:\n",
    "    with open(key_path, 'r') as file:\n",
    "        key = file.read().strip()\n",
    "except FileNotFoundError:\n",
    "        print(f\"Error: The file '{key_path}' was not found.\")\n",
    "except Exception as e:\n",
    "    print(f\"An error occured: {e}\")\n",
    "try:\n",
    "    with open(project_path, 'r') as file:\n",
    "        project = file.read().strip()\n",
    "except FileNotFoundError:\n",
    "        print(f\"Error: The file '{key_path}' was not found.\")\n",
    "except Exception as e:\n",
    "    print(f\"An error occured: {e}\")\n",
    "\n",
    "# just stop everything if the key or project id's not loaded here\n",
    "if not key or not project:\n",
    "    raise ValueError(\"Stopping execution: API Key or Project ID is missing/empty.\")\n",
    "\n",
    "os.environ[\"WATSONX_APIKEY\"] = key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "b411e04b-7c83-4f1e-8f24-8adfa7de410f",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "File path  is not a valid file or url",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[39], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# load pdf \u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m loader \u001b[38;5;241m=\u001b[39m \u001b[43mPyMuPDF4LLMLoader\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m      3\u001b[0m raw_docs \u001b[38;5;241m=\u001b[39m loader\u001b[38;5;241m.\u001b[39mload()\n\u001b[1;32m      5\u001b[0m \u001b[38;5;66;03m# split text \u001b[39;00m\n\u001b[1;32m      6\u001b[0m \u001b[38;5;66;03m# we use a large chunk size (1000) to keep medical concepts together\u001b[39;00m\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.12/site-packages/langchain_pymupdf4llm/pymupdf4llm_loader.py:228\u001b[0m, in \u001b[0;36mPyMuPDF4LLMLoader.__init__\u001b[0;34m(self, file_path, headers, password, mode, pages_delimiter, extract_images, images_parser, **pymupdf4llm_kwargs)\u001b[0m\n\u001b[1;32m    186\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Initialize with a file path.\u001b[39;00m\n\u001b[1;32m    187\u001b[0m \n\u001b[1;32m    188\u001b[0m \u001b[38;5;124;03mArgs:\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    221\u001b[0m \u001b[38;5;124;03m        `page_chunks`, `extract_words`, `show_progress`).\u001b[39;00m\n\u001b[1;32m    222\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    224\u001b[0m \u001b[38;5;66;03m# Input validation logic is primarily handled within the PyMuPDF4LLMParser,\u001b[39;00m\n\u001b[1;32m    225\u001b[0m \u001b[38;5;66;03m# so we don't need to repeat all checks here.\u001b[39;00m\n\u001b[1;32m    226\u001b[0m \u001b[38;5;66;03m# We pass the kwargs directly to the parser.\u001b[39;00m\n\u001b[0;32m--> 228\u001b[0m \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[38;5;21;43m__init__\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mfile_path\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mheaders\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mheaders\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    229\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mparser \u001b[38;5;241m=\u001b[39m PyMuPDF4LLMParser(\n\u001b[1;32m    230\u001b[0m     password\u001b[38;5;241m=\u001b[39mpassword,\n\u001b[1;32m    231\u001b[0m     mode\u001b[38;5;241m=\u001b[39mmode,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    235\u001b[0m     \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mpymupdf4llm_kwargs,\n\u001b[1;32m    236\u001b[0m )\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.12/site-packages/langchain_pymupdf4llm/pymupdf4llm_loader.py:82\u001b[0m, in \u001b[0;36mBasePDFLoader.__init__\u001b[0;34m(self, file_path, headers)\u001b[0m\n\u001b[1;32m     80\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mstr\u001b[39m(temp_pdf)\n\u001b[1;32m     81\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39misfile(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_path):\n\u001b[0;32m---> 82\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFile path \u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m is not a valid file or url\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_path)\n",
      "\u001b[0;31mValueError\u001b[0m: File path  is not a valid file or url"
     ]
    }
   ],
   "source": [
    "# load pdf \n",
    "loader = PyMuPDF4LLMLoader(\"\")\n",
    "raw_docs = loader.load()\n",
    "\n",
    "# split text \n",
    "# we use a large chunk size (1000) to keep medical concepts together\n",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)\n",
    "splits = text_splitter.split_documents(raw_docs)\n",
    "\n",
    "vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)\n",
    "retriever = vectorstore.as_retriever(search_kwargs = {\"k\": 4})\n",
    "\n",
    "# the citation formater\n",
    "def format_docs_with_sources(docs):\n",
    "    formatted_text = \"\"\n",
    "    for doc in docs:\n",
    "        # extract metadata\n",
    "        soruce = doc.metadata.get(\"source\", \"Unknown File\")\n",
    "        page = doc.metadata.get(\"page\", \"Unkown Page\")\n",
    "\n",
    "        # modify the content to physically include source ID like [Source: medical_study.pdf, Page 2]\n",
    "        content = doc.page_content.replace(\"\\n\", \" \")\n",
    "        formatted_text += f\"Content: {content}\\nSource: [{source}, Page {page}]\\n\\n\"\n",
    "    return formatted_text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "52a06fcd-7ca4-4066-b897-dcb60dd02aa6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os \n",
    "from langchain_ibm import WatsonxEmbeddings, ChatWatsonx\n",
    "from langchain_chroma import Chroma\n",
    "from langchain_pymupdf4llm import PyMuPDF4LLMLoader\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "from langchain_core.runnables import RunnablePassthrough\n",
    "\n",
    "embeddings = WatsonxEmbeddings(\n",
    "    model_id = \"ibm/granite-embedding-278m-multilingual\",\n",
    "    url=\"https://us-south.ml.cloud.ibm.com\",\n",
    "    project_id=project,\n",
    "    params={\"decoding_method\": \"greedy\"} \n",
    ")\n",
    "\n",
    "parameters = {\n",
    "    \"temperature\": 0.6,\n",
    "    \"max_tokens\": 1000, #since medical answers and synthesis tend to be on the longer side\n",
    "}\n",
    "\n",
    "chat = ChatWatsonx(\n",
    "    model_id = \"ibm/granite-3-3-8b-instruct\",\n",
    "    url = \"https://us-south.ml.cloud.ibm.com\",\n",
    "    project_id = project,\n",
    "    params = parameters,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f074f4cb-4ba2-466b-8557-b2af8908a118",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1147c3f6-9c97-4483-8a47-26c68e6458d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "\n",
    "# how granite would react to your messages and such\n",
    "system_instruct = \"\"\"You are an expert research assistant. Your task is to answer the user's question by synthesizing information from provided documents.\n",
    "\n",
    "Strict Rules:\n",
    "1. Grounding: Base your answer exclusively on the information contained in the \"Provided Documents\" section. Do not use any external knowledge or make assumptions.\n",
    "2. Citation: You must cite the source for every fact or claim in your answer. Use the format [DOC_ID] at the end of the sentence or paragraph that uses the information.\n",
    "3. Synthesis: If the question requires information from multiple documents, you must synthesize these pieces of information into a single, coherent answer.\n",
    "4. Unknown Information: If the documents do not contain enough information to answer the question, you must state: \"The provided documents do not contain information to answer this question.\" Do not try to guess.\"\"\"\n",
    "\n",
    "# for the template we're gonna get the context (which would be the docs) and the {question} (the user input)\n",
    "pompt = ChatPromptTemplate.from_template(system_instruct)\n",
    "\n",
    "# this makes it so it has the context, docs with sources \n",
    "rag_chain = (\n",
    "    {\"context\": retriever | format_docs_with_sources, \"questions\": RunnablePassthrough()}\n",
    "    | prompt\n",
    "    | chat\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6a27217e-8b16-4fd6-bc7b-e0fafb2fd42e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "---INPUT---\n",
      "Context:\n",
      "[DOC_1] The Granite model is developed by IBM\n",
      "[DOC_2] IBM's headquarters are located in Armonk, New York\n",
      "[DOC_3] Python's a programming language released in 1991\n",
      "\n",
      "Question: who developed Granite and where are they located?\n",
      "\n",
      "---OUTPUT---\n",
      "bThe Granite model was developed by IBM. The location of IBM's headquarters is Armonk, New York [DOC_1, DOC_2]. Unfortunately, there is no information provided about the specific location where the Granite model was developed within IBM.\n"
     ]
    }
   ],
   "source": [
    "#this is a mock area for testing\n",
    "doc1 = \"[DOC_1] The Granite model is developed by IBM\"\n",
    "doc2 = \"[DOC_2] IBM's headquarters are located in Armonk, New York\"\n",
    "doc3 = \"[DOC_3] Python's a programming language released in 1991\"\n",
    "\n",
    "context_text = f\"{doc1}\\n{doc2}\\n{doc3}\"\n",
    "\n",
    "question_text = \"who developed Granite and where are they located?\" #should use doc 1 and 2 (hopefully)\n",
    "\n",
    "print(f\"---INPUT---\\nContext:\\n{context_text}\\n\\nQuestion: {question_text}\\n\")\n",
    "\n",
    "response = rag_chain.invoke({\n",
    "    \"context\" : context_text,\n",
    "    \"question\" : question_text\n",
    "})\n",
    "\n",
    "print(f\"---OUTPUT---\\nb{response.content}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "268d1197-a5e3-4553-a34c-ed5ca03db77a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
