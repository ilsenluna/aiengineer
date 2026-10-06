from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1", temperature=0)
parser = StrOutputParser()

generate_large_text = (
    ChatPromptTemplate.from_template("Write one informative paragraph of about 150 words about: {topic}") 
    | model 
    | parser
)

format = RunnableLambda(lambda t: [l.strip().lstrip("-•* ").strip() for l in t.splitlines() if l.strip()][:3])
summarize = (
    ChatPromptTemplate.from_template("Summarize exactly in three bullet points, a bullet per line, no introduction or conclusion. This is the text: {text}")
    | model
    | parser
    | format
)

pipeline = (
    RunnablePassthrough.assign(text=generate_large_text) |
    RunnablePassthrough.assign(summary=summarize)
)

# tools
@tool
def explain_topic(topic: str) -> str:
    """Generate an informative paragraph and a three-bullet summary about a topic.
    Use it when the user asks to explain or learn about something."""
    result = pipeline.invoke({"topic": topic})
    bullets = "\n".join(f"- {b}" for b in result["summary"])
    return f"{result['text']}\n\nSummary:\n{bullets}"

@tool
def count_words(text: str) -> str:
    """Count the number of words in a text."""
    return len(text.split())

tools = [explain_topic, count_words]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant."
               "Use the tools when they help answer the question, and use the history to remember what the user has told you."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{pregunta}"),
])


def run_with_tools(inputs: dict, max_steps: int = 5) -> str:
    messages = prompt.invoke(inputs).to_messages()

    for _ in range(max_steps):
        ai = model_with_tools.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            return ai.content

        for call in ai.tool_calls:
            try:
                tool_msg = tools_by_name[call["name"]].invoke(call) 
            except Exception as e:
                tool_msg = ToolMessage(content=f"Tool error: {e}", tool_call_id=call["id"])
            messages.append(tool_msg)

    return "I was unable to complete the application."


cadena = RunnableLambda(run_with_tools)

# memory
records: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in records:
        records[session_id] = InMemoryChatMessageHistory()
    return records[session_id]


chat = RunnableWithMessageHistory(
    cadena,
    get_history,
    input_messages_key="pregunta",
    history_messages_key="historial",
)

if __name__ == "__main__":
    # Part 1: original chain
    result = pipeline.invoke({"topic": "What is LangChain"})
    print()
    print("STEP 1: Large text generated")
    print()
    print(result["text"])
    print()
    print("STEP 2: Summary")
    print()
    for i in result["summary"]:
        print(f"  • {i}")

    # Part 2: conversation with per-session memory + tools
    print()
    print("STEP 3: Conversation with memory and tools")
    print()
    config = {"configurable": {"session_id": "student-1"}}
    turnos = [
        "Hi, my name is Ilsen.",
        "Explain what LangChain is.",
        "What is my name and what topic did I ask you about?",
        "How many words does this sentence have: 'memory makes chains useful'?",
    ]
    for i, pregunta in enumerate(turnos, 1):
        respuesta = chat.invoke({"pregunta": pregunta}, config=config)
        print(f"[Turn {i}] You: {pregunta}")
        print(f"[Turn {i}] Bot: {respuesta}\n")

    # Control test: another session does NOT share memory
    otra = chat.invoke(
        {"pregunta": "What is my name?"},
        config={"configurable": {"session_id": "student-2"}},
    )
    print(f"[Other session] Bot: {otra}")
    print(f"\nMessages stored in 'student-1': {len(records['student-1'].messages)}")