from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
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

if __name__ == "__main__":
    result = pipeline.invoke({"topic": "What is LangGraph?"})
    print()
    print("STEP 1: Large text generated")
    print()
    print(result["text"])
    print()
    print("STEP 2: Summary")
    print()
    for i in result["summary"]:
        print(f"  • {i}")