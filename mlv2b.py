from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection
import uuid

import os
from dotenv import load_dotenv, find_dotenv

from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# import
#from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pymilvus import MilvusClient, DataType
from langchain_core.documents import Document
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient

MODEL = "gpt-4o-mini"

# Carga las variables de entorno desde un archivo .env
load_dotenv(find_dotenv(), override=True)

# Obtiene la API key de OpenAI desde las variables de entorno
api_key_openAI = os.environ.get("OPENAI_API_KEY")
print(api_key_openAI)

# Crear la función de embeddings de código abierto
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

uri = "http://localhost:19530"

# Conectar al cliente Milvus
client = MilvusClient(
    uri=uri,
    user="root",
    password="Milvus",
    token="joaquin:chamorro",
)

connections.connect(
    host="localhost",
    port="19530",
)

db_name = "my_database6"
if db_name not in db.list_database():
    print("database not found")
else:
    db.using_database(db_name)
    print(f"Conectado a la base de datos {db_name}")

# Crear una colección en Milvus
collection_name = "joaquin_DB2"

database = client.load_collection(collection_name)

vector_store = Milvus(
    embeddings,
    collection_name=collection_name,
    connection_args={"uri": uri}
)

#query = "Tratar la enfermedad de parkinson"
query = "Tabla 15–1. Clasificación general de movimientos anormales."

model = ChatOpenAI(api_key=api_key_openAI, model=MODEL)

retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 10, "filter": {"chapter": "30"}})
    #search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})


template =  """
            - Contesta como un profesional medico: {context}
            - Si no se aportan documentos:
                - Menciona que no se aportan documentos
                - Responde con tu conocimiento
            - Question: {question}
            """
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()})
chain = setup_and_retrieval | prompt | model | output_parser
respuesta=chain.invoke(query)

print(respuesta)

