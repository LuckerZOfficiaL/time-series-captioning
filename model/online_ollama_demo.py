from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat
from llm_axe import OnlineAgent, OllamaChat

def main():
    llm = OllamaChat(model="llama3.3")
    online_agent = OnlineAgent(llm)

    prompt = f"""
    Is this statement true? 
    \n
    There has been a female president in the US.
    \n

    You answer should be just "yes" or "no", without adding any extra text or explanation.
    """
    resp = online_agent.search(prompt)
    print(f"{resp}\n Time: {(end_time-start_time):.4f}")

if __name__ == "__main__":
    main()