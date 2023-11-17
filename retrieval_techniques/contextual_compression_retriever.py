from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from decouple import config

Text = ["Python is dynamically typed general purpose programming language.",
         "It is programming language used for data science.",
         "Python is also used in machine learning.",
         "It is used often also used in deeplearning.",
         "Python is used for OpenCv and neural network."
]

meta_data = [{"source": "document1", "page": 1},
             {"source": "document2", "page": 2},
             {"source": "document3", "page": 3},
             {"source": "document4", "page": 4},
             {"source": "document5", "page": 5},
]

embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts = Text,
    embedding = embedding_function,
    metadatas = meta_data
)

llm = OpenAI(temperature = 0, openai_api_key = config("OPENAI_API_KEY"))
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor =compressor,
    base_retriever = vector_db.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents("Tell me about programming language used for data science")
print(compressed_docs)