"""
ReAct Agent with LangGraph and OpenRouter
Manual tool orchestration for free models without native function calling.
"""
import os
import json
import warnings
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Suppress specific warnings from langchain_tavily
warnings.filterwarnings("ignore", message="Field name.*shadows an attribute")

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - Change model here (matches main.py)
# ============================================================================
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# Alternative models to try:
# MODEL_NAME = "google/gemini-2.0-flash-exp:free"
# MODEL_NAME = "nousresearch/hermes-3-llama-3.1-405b:free"
# MODEL_NAME = "mistralai/mistral-7b-instruct:free"
# MODEL_NAME = "qwen/qwen-2-7b-instruct:free"

TEMPERATURE = 0  # 0 = deterministic, 1 = creative
MAX_ITERATIONS = 5  # Maximum agent loop iterations
# ============================================================================


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


class AgentState(TypedDict):
    """State of the agent throughout the workflow."""
    messages: Annotated[list, add_messages]
    iteration: int
    sources: List[str]  # Track source URLs throughout the workflow


def create_openrouter_llm(model: str = MODEL_NAME, temperature: float = TEMPERATURE):
    """
    Create a ChatOpenAI instance configured to use OpenRouter.
    
    Args:
        model: The model to use via OpenRouter
        temperature: Temperature setting for the LLM (0 = deterministic)
        
    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/blurred421/langchain-course",
            "X-Title": "LangChain Course - ReAct Agent",
        }
    )


def create_react_prompt():
    """
    Create a ReAct-style prompt that instructs the model to use structured output.
    
    Returns:
        ChatPromptTemplate for the agent
    """
    system_prompt = """You are a helpful AI assistant that can search the web for information.

AVAILABLE TOOLS:
1. tavily_search: Search the web for current information
   - Input: A search query string
   - Use this when you need to find current information, job postings, news, etc.

INSTRUCTIONS:
When you need to use a tool, respond with a JSON object in this EXACT format:
{{
  "action": "tavily_search",
  "action_input": "your search query here"
}}

When you have enough information to answer the user's question, respond with a JSON object:
{{
  "action": "final_answer",
  "action_input": "your final answer here"
}}

IMPORTANT:
- Your response must be ONLY the JSON object, nothing else
- Do not add any explanation before or after the JSON
- If you need to search multiple times, do one search at a time
- After receiving search results, you can either search again or provide the final answer
- When providing the final answer, include specific details and information from the search results

REASONING PROCESS (ReAct):
1. Think about what information you need
2. Decide if you need to search or if you can answer
3. Output the appropriate JSON action
"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])


def llm_node(state: AgentState) -> AgentState:
    """
    LLM reasoning node - decides what action to take.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with LLM response
    """
    print(f"\n{'='*70}")
    print(f"Iteration {state['iteration'] + 1}/{MAX_ITERATIONS}")
    print(f"{'='*70}")
    
    llm = create_openrouter_llm()
    prompt = create_react_prompt()
    chain = prompt | llm
    
    response = chain.invoke({"messages": state["messages"]})
    
    print(f"\nLLM Response:\n{response.content[:200]}...\n")
    
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }


def parse_action(state: AgentState) -> Literal["tool", "end"]:
    """
    Parse the LLM's response to determine next action.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute: "tool" or "end"
    """
    last_message = state["messages"][-1]
    
    try:
        # Try to parse JSON from the response
        content = last_message.content.strip()
        
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        action_data = json.loads(content)
        action = action_data.get("action", "")
        
        print(f"Parsed Action: {action}")
        
        if action == "final_answer":
            return "end"
        elif action == "tavily_search":
            return "tool"
        else:
            print(f"Unknown action: {action}, ending.")
            return "end"
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Content was: {last_message.content[:200]}")
        # If we can't parse, assume it's a final answer
        return "end"


def extract_urls_from_tavily_results(results: str) -> List[str]:
    """
    Extract URLs from Tavily search results.
    
    Args:
        results: String containing search results
        
    Returns:
        List of URLs found in the results
    """
    urls = []
    try:
        # Tavily results are typically formatted with 'url': 'https://...'
        import re
        url_pattern = r"'url':\s*'([^']+)'"
        matches = re.findall(url_pattern, results)
        urls.extend(matches)
    except Exception as e:
        print(f"Error extracting URLs: {e}")
    
    return urls


def tool_node(state: AgentState) -> AgentState:
    """
    Execute the tool based on parsed action.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool results
    """
    last_message = state["messages"][-1]
    
    try:
        content = last_message.content.strip()
        
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        action_data = json.loads(content)
        action = action_data.get("action", "")
        action_input = action_data.get("action_input", "")
        
        if action == "tavily_search":
            print(f"\nExecuting Tavily Search: {action_input}")
            search_tool = TavilySearch(max_results=5)
            results = search_tool.invoke({"query": action_input})
            
            # Extract URLs from results
            new_urls = extract_urls_from_tavily_results(str(results))
            current_sources = state.get("sources", [])
            updated_sources = current_sources + new_urls
            
            observation_message = HumanMessage(
                content=f"Search results for '{action_input}':\n\n{results}\n\nBased on these results, either provide the final answer or search for more information if needed."
            )
            
            print(f"\nSearch Results Received (length: {len(str(results))} chars)")
            print(f"Extracted {len(new_urls)} URLs")
            
            return {
                "messages": [observation_message],
                "sources": updated_sources
            }
        else:
            return {"messages": [HumanMessage(content="Unknown tool action.")]}
            
    except Exception as e:
        error_message = HumanMessage(content=f"Error executing tool: {str(e)}")
        return {"messages": [error_message]}


def create_agent_graph():
    """
    Create the LangGraph agent workflow.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("tool", tool_node)
    
    # Set entry point
    workflow.set_entry_point("llm")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "llm",
        parse_action,
        {
            "tool": "tool",
            "end": END
        }
    )
    
    # Tool always goes back to LLM
    workflow.add_edge("tool", "llm")
    
    return workflow.compile()


def run_agent(query: str) -> AgentResponse:
    """
    Run the agent with the given query.
    
    Args:
        query: User's question/request
        
    Returns:
        AgentResponse with answer and sources
    """
    print(f"\n{'='*70}")
    print("ReAct Agent with LangGraph")
    print(f"{'='*70}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Query: {query}")
    
    agent = create_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iteration": 0,
        "sources": []
    }
    
    result = agent.invoke(initial_state)
    
    # Extract final answer
    last_message = result["messages"][-1]
    answer = ""
    
    try:
        content = last_message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        final_data = json.loads(content)
        if final_data.get("action") == "final_answer":
            answer = final_data.get("action_input", "")
        else:
            answer = content
    except:
        # If not JSON, just use the content
        answer = last_message.content
    
    # Create AgentResponse with sources
    sources = [Source(url=url) for url in result.get("sources", [])]
    agent_response = AgentResponse(answer=answer, sources=sources)
    
    return agent_response


def main():
    """Main entry point for the ReAct agent application."""
    print("Hello from langchain-course!\n")
    
    query = "Search for 3 job postings for an AI engineer using langchain in the Pensacola Florida and surrounding area on linkedin and list their details"
    
    response = run_agent(query)
    
    print(f"\n{'='*70}")
    print("FINAL RESULT")
    print(f"{'='*70}")
    print(f"\nAnswer:\n{response.answer}\n")
    
    if response.sources:
        print(f"Sources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. {source.url}")
    else:
        print("No sources found.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()