

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from colorama import Fore
from groq import Groq

from agents_pattern.utils.completions import (
    build_prompt_structure,
    chat_completions_create,
    update_chat_history,
    FixedFirstChatHistory 
)

from agents_pattern.utils.logging import fancy_step_tracker
from agents_pattern.reflection.reflection_prompt import BASE_REFLECTION_SYSTEM_PROMPT, BASE_GENERATION_SYSTEM_PROMPT
from agents_pattern.settings import settings



class ReflectionAgent:
    """
    A class that implements a Reflection Agent, which generates responses and reflects 
    on them using the LLM to iteratively improve the interaction. The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.

    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (Groq): An instance of the Groq client to interact with the language model.
    """

    CLIENT = Groq(api_key=settings.GROQ_API_KEY)

    @classmethod
    def request_chat_completion(cls, history: list, verbose: int = 0,log_title: str = "COMPLETION", log_color: str = "",):
        """
        Requests a chat completion from the language model and updates the chat history.

        Args:
            history (list): The chat history.
        """

        output = chat_completions_create(client= cls.CLIENT, messages=history, model= settings.GROQ_TEXT_MODEL_NAME)
        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)
        
        return output
    
    @classmethod
    def generate_response(cls, generation_history: list, verbose: int = 0) -> str:
        """
        Generates a response based on the provided generation history.

        Args:
            generation_history (list): The generation history.

        Returns:
            str: The generated response.
        """
        return cls.request_chat_completion(history=generation_history, verbose=verbose, log_title="GENERATION", log_color=Fore.BLUE)
    
    @classmethod
    def reflect_response(cls, reflection_history: list, verbose: int = 0) -> str:
        """
        Reflects on the provided reflection history.

        Args:
            reflection_history (list): The reflection history.

        Returns:
            str: The reflected response.
        """
        return cls.request_chat_completion(history=reflection_history, verbose=verbose, log_title="REFLECTION", log_color=Fore.GREEN)
    
    @classmethod
    def run(cls, user_msg: str,n_steps: int,verbose:int = 0)->str:
        """
        Runs the ReflectionAgent over multiple steps, alternating between generating a response
        and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message to generate a response for.
            n_steps (int): The number of reflection steps to run.

        Returns:
            str: The final reflected response.
        """

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=BASE_GENERATION_SYSTEM_PROMPT, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3
        )

        reflection_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=BASE_REFLECTION_SYSTEM_PROMPT, role="system"),
            ],
            total_length=3
        )

        for step in range(n_steps):
            
            if verbose > 0:
                fancy_step_tracker(step, n_steps)
                
            # Generation 
            generation = cls.generate_response(generation_history, verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            # reflection 
            reflection = cls.reflect_response(reflection_history, verbose)

            if '<OK>' in reflection:
                print(
                    Fore.RED,
                    "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                )
                break

            update_chat_history(generation_history, reflection, "user")
            update_chat_history(reflection_history, reflection, "assistant")

        return generation
    
agent = ReflectionAgent()
user_msg = "write a poem on Chennai Super Kings IPL team in 4 lines"

response = agent.run(
    user_msg=user_msg,
    n_steps=6,
    verbose=1
)

print(response)