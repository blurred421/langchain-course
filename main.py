"""
LangChain Course - OpenRouter Integration with LangChain
Example of using LangChain with OpenRouter API to generate summaries.
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - Change model here
# ============================================================================
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# Alternative models to try:
# MODEL_NAME = "google/gemini-2.0-flash-exp:free"
# MODEL_NAME = "nousresearch/hermes-3-llama-3.1-405b:free"
# MODEL_NAME = "mistralai/mistral-7b-instruct:free"
# MODEL_NAME = "qwen/qwen-2-7b-instruct:free"
# MODEL_NAME = "anthropic/claude-3-100k:free"
# MODEL_NAME = "openai/gpt-oss-120b:free"
# MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
# MODEL_NAME = "gpt-4o-mini:free"
# MODEL_NAME = "allenai/olmo-3.1-32b-think:free"


TEMPERATURE = 0  # 0 = deterministic, 1 = creative
# ============================================================================


def get_person_information() -> str:
    """
    Returns sample information about a person for demonstration.
    
    Returns:
        str: Biographical information
    """
    return """
    Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He received bachelor's degrees from the University of Pennsylvania in 1997 before moving to California, United States, to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    """


def create_summary_chain(model: str = MODEL_NAME, temperature: float = TEMPERATURE):
    """
    Creates a LangChain chain for generating person summaries.
    
    Args:
        model: The model to use via OpenRouter
        temperature: Temperature setting for the LLM (0 = deterministic)
        
    Returns:
        A LangChain chain that can process person information
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # Configure ChatOpenAI to use OpenRouter with required headers
    llm = ChatOpenAI(
        temperature=temperature,
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/blurred421/langchain-course",
            "X-Title": "LangChain Course",
        }
    )
    
    # Create the chain using LCEL (LangChain Expression Language)
    chain = summary_prompt_template | llm
    
    return chain


def generate_summary(information: str, model: str = MODEL_NAME) -> str:
    """
    Generate a summary and interesting facts about a person.
    
    Args:
        information: Biographical information about the person
        model: The model to use (default: configured MODEL_NAME)
        
    Returns:
        str: The generated summary and facts
    """
    chain = create_summary_chain(model=model)
    response = chain.invoke(input={"information": information})
    return response.content


def main():
    """Main entry point for the application."""
    print("Hello from langchain-course!\n")
    print(f"Using model: {MODEL_NAME}\n")
    
    # Get sample information
    information = get_person_information()
    
    # Generate summary using OpenRouter with configured model
    print("Generating summary using OpenRouter...\n")
    summary = generate_summary(information)
    print(summary)


if __name__ == "__main__":
    main()