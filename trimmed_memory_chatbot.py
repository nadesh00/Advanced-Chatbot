from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import trim_messages
import os
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize trimmer - keeps only last 10 tokens (5 conversation pairs)
# Each message counts as 1 token using len as counter
trimmer = trim_messages(strategy="last", max_tokens=10, token_counter=len)

# Create the graph
builder = StateGraph(state_schema=MessagesState)

# Define chat node with message trimming
def chat_node(state: MessagesState):
    """
    Chatbot with message trimming.
    Only keeps the last 10 messages to save tokens and stay within context limits.
    """
    # Trim messages before processing
    trimmed_messages = trimmer.invoke(state["messages"])
    
    system_message = SystemMessage(content="You're a kind therapy assistant.")
    prompt = [system_message] + trimmed_messages
    response = model.invoke(prompt)
    
    return {"messages": response}

# Build graph
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")

# Compile graph with MemorySaver
memory = MemorySaver()
chat_app = builder.compile(checkpointer=memory)

# Unique thread ID for this conversation
thread_id = "2"

# Main conversation loop
print("Therapy Chatbot (Trimmed Memory - Last 10 messages)")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    state_update = {"messages": [HumanMessage(content=user_input)]}
    
    result = chat_app.invoke(
        state_update,
        {"configurable": {"thread_id": thread_id}}
    )
    
    ai_msg = result["messages"][-1]
    print(f"Bot: {ai_msg.content}\n")
    
    # Uncomment to see how many messages are actually stored
    # print(f"Messages in memory: {len(result['messages'])}")