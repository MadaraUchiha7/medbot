from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from decouple import config
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory = "../vector_db",
    embedding_function = embedding_function,
    collection_name = "GaleEncyclopediaofMedicine.pdf"
)

llm = ChatOpenAI(openai_api_key = config("OPENAI_API_KEY"), temperature = 0.5)

memory = ConversationBufferMemory(
    return_messages = True,
    memory_key = "chat_history"
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory = memory,
    retriever = vector_db.as_retriever(
        search_kwargs={"fetch_k": 4, "k": 3}, search_type = "mmr"
    ),
    chain_type = "refine"
)
def rag_func(question: str) -> str:
    response = qa_chain({"question": question})
    return response.get("answer")