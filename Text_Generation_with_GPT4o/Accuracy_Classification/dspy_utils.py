
import os
import random
from typing import Any
from dotenv import load_dotenv


import dspy
import dspy.teleprompt


class Predict(dspy.Module):

    def __init__(
            self, 
            signature: dspy.Signature,
            chain_of_thought: bool = False,
            temperature: float = 1.0,
            use_cache: bool = False,
            **config: Any,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            use_cache = True
        self.predict = (
            dspy.ChainOfThought(signature, temperature=temperature, **config) 
            if chain_of_thought 
            else dspy.Predict(signature, temperature=temperature, **config)
        )
        self.temperature = temperature
        self.use_cache = use_cache
        self.chain_of_thought = chain_of_thought

    def forward(self, **kwargs) -> dspy.Prediction:
        # The below is a hack to "disable" caching:
        # Predict only caches if both the call *and* the config have already appeared identically.
        # --> adding slight jitter to the temperature 
        if not self.use_cache:
            try:
                self.predict._predict.config["temperature"] = self.temperature + random.random() * 1e-5
            except AttributeError:
                self.predict.config["temperature"] = self.temperature + random.random() * 1e-5

        return self.predict(**kwargs)
    

def load_client(model: str = "gpt-4o", env_addon: str = "_SWEDEN") -> dspy.AzureOpenAI:
    load_dotenv()
    return dspy.AzureOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY" + env_addon),
        api_base=os.getenv("OPENAI_API_BASE" + env_addon),
        api_version=os.getenv("OPENAI_API_VERSION" + env_addon),
        max_tokens=4096,
    )


class OptimizablePredict(dspy.Module):
    def __init__(self, signature: dspy.Signature, **config: Any):
        super().__init__()
        self.predict = dspy.Predict(signature=signature, **config)

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
