"""
This module provides functionality for generating reasoning on recovery strategies
for reinforcement learning agents in out-of-distribution (OOD) scenarios using OpenAI's GPT-4 model.

Classes:
    BehaviorReasoner: Handles reasoning generation based on OOD descriptions and task prompts.

Functions:
    main(): Demonstrates the usage of the BehaviorReasoner class for generating reasoning outputs.
"""

import sys, os
from typing import Optional, Union
from openai import OpenAI
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import get_original_task

def get_args():
    parser = argparse.ArgumentParser(description='ood_descriptor')
    parser.add_argument(
        '--env',
        default='ant'
    )
    args = parser.parse_args()
    return args

class BehaviorReasoner:
    """
    A class to handle reasoning generation for recovery strategies in OOD scenarios
    using OpenAI's GPT-4 model.
    """
    
    def __init__(self, client=None, api_key_path: Optional[str] = "openai_key.txt"):
        """
        Initialize the BehaviorReasoner class with an OpenAI API client.

        Args:
            client (optional): An existing OpenAI client instance. If None, a new client 
                will be created using the API key from the specified file.
            api_key_path (str, optional): Path to the file containing the OpenAI API key. 
                This is used only if `client` is None. Defaults to "openai_key.txt".
        """
        if client is None:
            self.api_key = self._load_api_key(api_key_path)
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = client

    def _load_api_key(self, api_key_path: str) -> str:
        """
        Load the OpenAI API key from a file.

        Args:
            api_key_path (str): Path to the file containing the API key.

        Returns:
            str: The API key.

        Raises:
            FileNotFoundError: If the specified API key file does not exist.
        """
        try:
            with open(api_key_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at: {api_key_path}")

    def _load_file(self, filepath: str) -> str:
        """
        Load content from a text file.

        Args:
            filepath (str): Path to the text file.

        Returns:
            str: Content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
    
    def get_reasoning(
        self,
        prompt_file: str,
        ood_description: str = None,
        ood_description_file: str = None,
        output_file: Union[str, None] = None,
        original_task: str = "",
        env_name: str = "",
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate reasoning based on the OOD description and task prompt.

        Args:
            ood_description_file (str): Path to the file containing the OOD state description.
            prompt_file (str): Path to the file containing the prompt template.
            output_file (str, optional): Path where the response should be saved. If None,
                the response is not saved to a file. Defaults to None.
            original_task (str): Description of the agent's original task.
            env_name (str, optional): Name of the environment to replace in the prompt template.
                Defaults to an empty string.
            max_tokens (int, optional): Maximum number of tokens for the API response.
                Defaults to 500.

        Returns:
            Optional[str]: The generated reasoning output, or None if an error occurs.

        Raises:
            FileNotFoundError: If the description file or prompt file is not found.
            Exception: For other unexpected errors during processing.
        """
        try:
            if not ood_description:
                # Read the OOD description
                ood_description = self._load_file(ood_description_file)
                
            # Read the prompt template
            prompt_template = self._load_file(prompt_file)
            
            # Construct the final prompt
            prompt = prompt_template.replace("{ood_description}", ood_description).replace("{original_task}", original_task).replace("{env_name}", env_name)
            
            # Prepare the input for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are a logical reasoning assistant that provides clear, structured analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Save the response to a file if specified
            if output_file and response_text:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(response_text)
            
            return response_text

        except Exception as e:
            print(f"Error in get_reasoning: {e}")
            return None

def main():
    """
    Demonstrates the usage of the BehaviorReasoner class for generating reasoning
    outputs for OOD scenarios using GPT-4.
    """
    try:
        args = get_args()
        
        if args.env == 'pushchair':
            original_task = get_original_task('PushChairNormal-v1')
        else:
            original_task = get_original_task('AntNormal-v3')
        
        # Initialize the BehaviorReasoner
        generator = BehaviorReasoner()
        
        # Generate reasoning
        response = generator.get_reasoning(
            prompt_file="prompts/behavior_reasoning.txt",
            ood_description_file=f"outputs/{args.env}/ood_description_output.txt",
            output_file=f"outputs/{args.env}/behavior_reasoning_output.txt",
            original_task=original_task,
            env_name=f"{args.env.capitalize()} environment"
        )
        
        if response:
            print("Reasoning output:", response)
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
