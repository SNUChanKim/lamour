"""
This module provides functionality for generating Python code based on inputs from prompt files,
using OpenAI's GPT-4 model.

Classes:
    CodeGenerator: Handles the loading of prompt files, generating code from OpenAI's GPT-4, and saving the output.

Functions:
    main(): Demonstrates the usage of the CodeGenerator class for generating Python code.
"""

import os
from typing import Optional, Union
from openai import OpenAI
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ood_descriptor')
    parser.add_argument(
        '--env',
        default='ant'
    )
    args = parser.parse_args()
    return args

class CodeGenerator:
    """
    A class to handle Python code generation using OpenAI's GPT-4 model
    based on inputs from multiple text files.
    """

    def __init__(self, client=None, api_key_path: Optional[str] = "openai_key.txt"):
        """
        Initialize the CodeGenerator class with an OpenAI API client.

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
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")

    def get_generated_code(
        self,
        prompt_file: str,
        environment_description_file: str,
        behavior_reasoning: str = None,
        reasoning_file: str = None,
        output_file: Union[str, None] = None,
        env_name: str = "",
        few_shot_example=None,
        max_tokens: int = 3000
    ) -> Optional[str]:    
        """
        Combine the contents of the input files into a single prompt and generate Python code.

        Args:
            reasoning_file (str): Path to the reasoning file containing behavioral insights.
            environment_description_file (str): Path to the file describing the environment.
            prompt_file (str): Path to the file containing the prompt template.
            output_file (str, optional): Path to save the generated Python code. If None, code is not saved.
            env_name (str, optional): Name of the environment to include in the generated prompt.
                Defaults to an empty string.
            max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 3000.

        Returns:
            Optional[str]: The generated Python code, or None if an error occurs.

        Raises:
            FileNotFoundError: If any of the input files are not found.
            Exception: If an error occurs during file loading or API communication.
        """
        try:    
            # Read the behavior reasoning content
            if not behavior_reasoning:
                # Read the Behavior reasoning
                behavior_reasoning = self._load_file(reasoning_file)
                
            # Read the environment description content
            environment_description = self._load_file(environment_description_file)
            
            # Read the prompt template
            prompt_template = self._load_file(prompt_file)
            
            # Construct the final prompt
            prompt = (
                prompt_template
                .replace("{environment_description}", environment_description)
                .replace("{recovery_behavior}", behavior_reasoning)
                .replace("{env_name}", env_name)
            )
            
            if few_shot_example:
                print("Few-shot example loaded.")
                example = self._load_file(few_shot_example)
                prompt = prompt.replace("{example}", example)

            # Prepare the input for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that generates Python code."
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
            response_code = response.choices[0].message.content.strip()

            # Save the response to a file if specified
            if output_file and response_code:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(response_code)

            return response_code

        except Exception as e:
            print(f"Error in code_generation: {e}")
            return None
    


def main():
    """
    Demonstrates the usage of the CodeGenerator class for generating Python code
    from given input files.
    """
    try:
        args = get_args()
        
        few_shot_example=None
        if args.env == 'pushchair':
            few_shot_example = "prompts/example.txt"
        
        # Initialize the CodeGenerator with your OpenAI API key
        generator = CodeGenerator()

        # Generate Python code directly from input files
        generated_code = generator.get_generated_code(
            prompt_file="prompts/code_generation.txt",
            environment_description_file=f"env_description/{args.env}_env.txt",
            reasoning_file=f"outputs/{args.env}/behavior_reasoning_output.txt",
            output_file=f"outputs/{args.env}/generated_code.py",
            env_name=f"{args.env.capitalize()} environment",
            few_shot_example=few_shot_example
        )

        print(f"Code generation complete. Output saved to outputs/{args.env}/generated_code.py.")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
