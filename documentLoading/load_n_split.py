from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

FILE_PATH = "../documents/GaleEncyclopediaofMedicine.pdf"

# Create loader
loader = PyPDFLoader(FILE_PATH)

# split document
pages = loader.load_and_split()

# embedding functions
embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents = pages,
    embedding = embedding_function,
    persist_directory = "../vector_db",
    collection_name = "GaleEncyclopediaofMedicine.pdf"
)

# make persitant
vectordb.persist()