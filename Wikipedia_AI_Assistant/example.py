#######################################################################################################
###### Python Package Imports for this notebook                                                  ######
#######################################################################################################

# from langchain.document_loaders import WikipediaLoader  # Legacy import - no longer supported
# from langchain_community.document_loaders import WikipediaLoader  # Sunset package
import ssl

import certifi
import httpx
import wikipediaapi
from langchain_core.documents import Document
import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import DatabricksEmbeddings

from databricks_langchain import (
    ChatDatabricks,
    DatabricksEmbeddings,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)


#######################################################################################################
###### Config (Define LLMs, Embeddings, Vector Store, Data Loader specs)                         ######
#######################################################################################################

# DataLoader Config
query_terms = ["sport", "football", "soccer", "basketball","baseball", "track","swimming", "gymnastics"] #TODO: update to match your use case requirements
max_docs = 10 #TODO: recommend starting with a smaller number for testing purposes
WIKIPEDIA_USER_AGENT = "DrillingWikipedia/1.0 (mai.la@berkeley.edu)"
# On Databricks, change this to the CA file's path in that environment.
WIKIPEDIA_CA_FILE = "/Users/mai.la/.nov-certs/NOVRootCert.crt"

# Retriever Config
k = 2 # number of documents to return
EMBEDDING_MODEL = "databricks-bge-large-en" # Embedding model endpoint name


# LLM Config
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-8b-instruct" # Model Serving endpoint name; other option see "Serving" under AI/ML tab (e.g. databricks-gpt-oss-20b)


example_question = "What is the most popular sport in the US?"

 #######################################################################################################
 ###### Wikipedia Data Loader                                                                     ######
 #######################################################################################################

# Legacy loader call (query must be a string, so load each topic separately):
# docs = WikipediaLoader(query=query_terms, load_max_docs=max_docs).load() # Load in documents from Wikipedia takes about 10 minutes for 1K articles
# Trust public CAs and the NOV CA used by Zscaler HTTPS inspection.
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.load_verify_locations(cafile=WIKIPEDIA_CA_FILE)

wiki = wikipediaapi.Wikipedia(
    user_agent=WIKIPEDIA_USER_AGENT,
    language="en",
    transport=httpx.HTTPTransport(verify=ssl_context),
)

docs = []
for query in query_terms:
    search_results = wiki.search(query, limit=max_docs)
    for title, page in search_results.pages.items():
        docs.append(
            Document(
                page_content=page.text[:5000],
                metadata={
                    "title": title,
                    "summary": page.summary,
                    "source": page.fullurl,
                },
            )
        )

for doc in docs:
    print(doc.metadata["title"])

#######################################################################################################
###### FAISS Retriever: Using DBX embedding model                                                ###### #######################################################################################################

# Define the embeddings and the FAISS vector store
embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_MODEL) # Use to generate embeddings
vector_store = FAISS.from_documents(docs, embeddings)

# Example of how to invoke the vector store
results = vector_store.similarity_search(
    "What is the most popular sport in the US?",
    k=k
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

#######################################################################################################
###### LLM: Using DBX Foundation Model                                                           ###### #######################################################################################################

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

response = llm.invoke("What is the most popular sport in the US?")

print("\n",response.content)
