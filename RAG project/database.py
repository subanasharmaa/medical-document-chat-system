#Load the pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
docs = data.load()

#splitting 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       
    chunk_overlap=300      
)
chunks = splitter.split_documents(docs)

#Creating embeddings and storing in the chroma
embedding_model = OpenAIEmbeddings() 
vectorstore = Chroma.from_documents(
    documents = chunks, 
    persist_directory="Chroma_db",
    embedding_function = embedding_model
)