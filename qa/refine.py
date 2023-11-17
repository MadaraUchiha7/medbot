from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
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

QA_Prompt = PromptTemplate(
    template = """Use the following piece of context to answer the user question.
    Context: {text}
    Question: {question}
    Answer:""",
    input_variables = ["text", "question"]
)

llm = ChatOpenAI(openai_api_key = config("OPENAI_API_KEY"), temperature = 0)
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_db.as_retriever(),
    return_source_documents = True,
    chain_type = "refine",
)
question = "What area is Python mostly used?"
response = qa_chain({"query": question})
print(response["result"])
print("-----------------------------------------------------")
print(response["source_documents"])