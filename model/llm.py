from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()


def get_llm(model_name: str="gemini-2.5-flash",temp : float= 0.5) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        max_output_tokens=2048,
        top_p=0.95,
        top_k=40,
        stream=False,
    )