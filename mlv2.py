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


#conn = connections.connect(host="localhost", port=19530)

#database = db.create_database("my_database")

# load the document and split it into chunks
#loader = PyPDFDirectoryLoader("./folder")
loader = TextLoader("./parkinson_gpt.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(
    separator="\n\n", 
    chunk_overlap=0
    )
docs = text_splitter.split_documents(documents)


# Crear la función de embeddings de código abierto
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

uri = "http://localhost:19530"

# Conectar al cliente Milvus
client = MilvusClient(
    uri=uri,
    token="joaquin:chamorro",
)

connections.connect(
    host="localhost",
    port="19530",
)

db_name = "my_database6"
if db_name not in db.list_database():
    db.create_database(db_name)
else:
    db.using_database(db_name)

print(f"Conectado a la base de datos {db_name}")

# Crear una colección en Milvus
collection_name = "joaquin_DB2"
dimension = 3072  # Especifica la dimensión del vector

if collection_name not in client.list_collections():
    fields = [
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    schema = CollectionSchema(fields=fields, description="Schema for quick_setup2")
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )


vector_store = Milvus.from_documents(
    docs,
    embeddings,
    collection_name=collection_name,
    connection_args={"uri": uri}
)

query = "Tratar la enfermedad de parkinson"

model = ChatOpenAI(api_key=api_key_openAI, model=MODEL)

retriever = vector_store.as_retriever(
    #search_type="similarity", search_kwargs={"k": 10, "filter": {"chapter": "30"}})
    search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})


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

