import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceEndpoint

load_dotenv()

class LangchainHealthAdvisor:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature=0.75, max_new_tokens=150):
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found in .env file.")
        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task="text-generation",
            headers={"Authorization": f"Bearer {hf_token}"}
        )
        
    def get_advice(self, prompt):
        response = self.llm(prompt)
        return response
