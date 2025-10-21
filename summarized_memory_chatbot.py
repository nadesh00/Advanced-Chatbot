from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
import os
from dotenv import load_dotenv
# Set your OpenAI API key
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create graph
builder = StateGraph(state_schema=MessagesState)

def chat_node(state: MessagesState):
    """
    Chatbot with summarization.
    When conversation exceeds 8 messages, summarizes old messages 
    and keeps only summary + recent messages.
    """
    system_message = SystemMessage(content="You're a kind therapy assistant.")
    
    # Get history excluding the latest message
    history = state["messages"][:-1]
    
    # If history exceeds 8 messages, summarize
    if len(history) >= 8:
        # Get the latest human message
        last_human_message = state["messages"][-1]
        
        # Create summary prompt
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        
        # Generate summary
        summary_message = model.invoke(history + [HumanMessage(content=summary_prompt)])
        
        # Create delete operations for old messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        
        # Recreate the human message
        human_message = HumanMessage(content=last_human_message.content)
        
        # Generate response with summary + current message
        response = model.invoke([system_message, summary_message, human_message])
        
        # Return summary, current message, response, and delete old messages
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        # For short conversations, use normal message chaining
        message_updates = model.invoke([system_message] + state["messages"])
    
    return {"messages": message_updates}

# Build graph
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")

# Compile graph with MemorySaver
memory = MemorySaver()
chat_app = builder.compile(checkpointer=memory)

# Unique thread ID
thread_id = "3"

# Main conversation loop
print("Therapy Chatbot (Summarization Memory - Summarizes after 8 messages)")
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
    
    # Uncomment to see full message structure (includes summaries)
    # print(f"Full result: {result}\n")