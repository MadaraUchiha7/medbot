from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
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
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="This is the source documents there are 4 main documents,  `document1`, `document2`, `document3`, `document4`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the details of Python",
        type="integer",
    ),
]

document_content_description = "Info on Python Programming Language"
llm = OpenAI(temperature = 0, openai_api_key = config("OPENAI_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)


docs = retriever.get_relevant_documents("What was mentioned in the 4th document about  Python")
print(docs)