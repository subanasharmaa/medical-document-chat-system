from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage #For history
from dotenv import load_dotenv
load_dotenv()

#VectorStore
embedding_model = OpenAIEmbeddings() 
vectorstore = Chroma(
    persist_directory="Chroma_db",
    embedding_function = embedding_model
)
#Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,            
        "fetch_k": 15,     
        "lambda_mult": 0.4 #0 diverse 1 less diverse
    }
)  
    
#Calling LLM

llm = ChatMistralAI(model = "mistral-small-2506")

#promptTemplate and chat history
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a medical education assistant designed to help 
medical students and healthcare learners. 

Use ONLY the provided context from the medical PDF to answer questions.
Follow these rules strictly:
- If the answer is not in the context, say: 'I could not find this in the provided medical document.'
- Always recommend consulting a licensed medical professional for real clinical decisions.
- Do not provide personal medical diagnoses or treatment advice.
- Use clear, precise medical terminology but explain terms when needed.
- If a question involves an emergency, always say: 'Please call emergency services immediately.' """),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", """Context: {context}

Question: {question}""")
])

chat_history = [] 
def get_response(query: str) -> str:
    # Retrieve relevant docs
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build & invoke prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
        "chat_history": chat_history                 
    })

    # Get response
    response = llm.invoke(final_prompt)

    # Update history
    chat_history.append(HumanMessage(content=query))        
    chat_history.append(AIMessage(content=response.content))  

    return response.content

print("Rag system is created.")
print("Press 0 to exit.")

while True:
    query = input("You: ").strip()
    if query == "0":
        print("Goodbye!")
        break
    if not query:
        continue
    answer = get_response(query)
    print(f"\nAI: {answer}\n")