from enum import Enum
from pathlib import Path
from typing import TypedDict

from langgraph.graph import START, END, StateGraph

class TypeMessage(str, Enum):
    QUESTION = "Question"
    COMPLAINT = "Complaint"


# State
class State(TypedDict):
    user_message: str


# Nodes
def classify_message(state: State) -> TypeMessage:
    message = state["user_message"]
    if "?" in message.lower():
        return TypeMessage.QUESTION
    else:
        return TypeMessage.COMPLAINT

# Edges
builder = StateGraph(State)
builder.add_node("classify_message", classify_message)
builder.add_edge(START, "classify_message")
builder.add_edge("classify_message", END)

# Graph
graph = builder.compile()

if __name__ == "__main__":
    print("=== First Testing the StateGraph ===")
    message_type = graph.invoke({"user_message": "What is the status of my order?"})
    #print(f"User Message: ", {"user_message": "What is the status of my order?"})
    #print(f"Message Type Result: {message_type.value}")  # Output: Message Type: Question

    #print("\n=== Second Testing the StateGraph ===")
    #message_type = graph.invoke({"user_message": "I am not happy with the service."})
    #print(f"User Message: ", {"user_message": "I am not happy with the service."})
    #print(f"Message Type Result: {message_type.value}")  # Output: Message Type: Complaint

    # Draw the graph
    #png = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    #with open("classify_message.png", "wb") as f:
    #    f.write(png)
    #print("Diagram classify_message.png saved successfully.")