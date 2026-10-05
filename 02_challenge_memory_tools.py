from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1", temperature=0)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Respond in Spanish and in one sentences."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{question}"),
])

if __name__ == "__main__":
    print("Llega aqui")