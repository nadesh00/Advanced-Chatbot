# LangChain Memory Chatbots - Trimming vs Summarization

This project implements two different approaches to managing conversational memory in LangChain chatbots using LangGraph.

## The Problem

LLMs have token limits and longer conversations become expensive. These chatbots solve that by managing memory differently.

---

## Trimming Chatbot (`trimmed_memory_chatbot.py`)

### How It Works
Keeps only the **last N messages** (default: 10) in memory. Older messages are ignored.

```python
trimmer = trim_messages(strategy="last", max_tokens=10, token_counter=len)
trimmed_messages = trimmer.invoke(state["messages"])
```

### Behavior
- **Turn 1-5**: Remembers everything (only 10 messages total)
- **Turn 6+**: Forgets messages beyond the 10-message window
- **Example**: If you say "I'm John" in turn 1, by turn 11 the bot forgets your name

### Pros
- Fast and simple
- Predictable token usage
- No extra LLM calls
- Low cost

### Cons
- Loses important context
- Arbitrary cutoff point
- Not intelligent about what to keep

### Best For
- Short conversations
- Transactional chatbots
- FAQ bots
- Budget-constrained apps

---

## Summarization Chatbot (`summarized_memory_chatbot.py`)

### How It Works
When conversation exceeds 8 messages, it **summarizes** old messages into a single compressed message, then deletes the originals.

```python
if len(history) >= 8:
    # Create summary of all old messages
    summary_message = model.invoke(history + [summary_prompt])
    
    # Delete old messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
    
    # Continue with: [summary] + [current message]
```

### Behavior
- **Turn 1-4**: Normal conversation (< 8 messages)
- **Turn 5+**: Summarizes turns 1-4 into one message
- **Example**: "John from SF, software engineer, feeling stressed about work..."
- **Result**: Bot remembers "John" even after 20 turns because it's in the summary

### Pros
- Maintains long-term context
- Remembers important details
- Scales to longer conversations
- More intelligent than trimming

### Cons
- Extra LLM calls for summaries (costs more)
- Summary quality varies
- If summary misses details, they're lost forever
- Slower response time

### Best For
- Long conversations (therapy, coaching)
- User support with context
- Personal assistants
- Production apps

---

## Comparison

| Feature | Trimming | Summarization |
|---------|----------|---------------|
| **Memory Type** | Last N messages | Compressed summary |
| **Context Retention** | Short-term only | Long-term |
| **Token Usage** | Very low | Moderate |
| **Extra LLM Calls** | None | Yes (for summaries) |
| **Cost** | Cheapest | Moderate |
| **Speed** | Fastest | Slower |
| **Complexity** | Simple | Moderate |
| **Best For** | Quick chats | Long conversations |

---

## Installation

```bash
# Install dependencies
pip install langchain langchain-openai langgraph python-dotenv

# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
echo ".env" > .gitignore
```

## Usage

```bash
# Run trimming chatbot
python trimmed_memory_chatbot.py

# Run summarization chatbot
python summarized_memory_chatbot.py
```

## Example Conversations

### Trimming (Loses Context)
```
You: Hi I'm John from SF
Bot: Hi John! Nice to meet you.

[... 10 more exchanges ...]

You: What's my name?
Bot: I don't have access to your name.
```
**Why**: Message with "John" was pushed out of the 10-message window.

### Summarization (Keeps Context)
```
You: Hi I'm John from SF
Bot: Hi John! Nice to meet you.

[... 10 more exchanges ...]

You: What's my name?
Bot: Your name is John!
```
**Why**: "John from SF" was preserved in the summary.

---

## When to Use Which?

**Use Trimming if:**
- Conversations are naturally short (< 10 exchanges)
- Users ask independent questions
- Budget is very tight
- Speed is critical

**Use Summarization if:**
- Conversations are long and personal
- Context from earlier matters
- Users expect the bot to "remember" them
- You're building for production

---
