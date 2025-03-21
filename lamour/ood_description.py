"""
This module provides functionality for processing images using OpenAI's GPT-4 Vision model
to generate detailed descriptions of out-of-distribution (OOD) states for reinforcement 
learning agents.

Classes:
    OODDescriptor: Handles image processing and interaction with the GPT-4 Vision API.

Functions:
    main(): Demonstrates the usage of the OODDescriptor class for generating OOD descriptions.
"""

import os
import base64
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

class OODDescriptor:
    """
    A class to handle image processing and description generation for 
    out-of-distribution (OOD) scenarios using OpenAI's GPT-4 Vision model.
    """
    
    def __init__(self, client=None, api_key_path: Optional[str] = "openai_key.txt"):
        """
        Initialize the OODDescriptor class with an OpenAI API client.

        Args:
            client (optional): An existing OpenAI client instance. If None, a new client 
                will be initialized using the provided API key.
            api_key_path (str, optional): Path to the file containing the OpenAI API key. 
                Used only if `client` is None. Defaults to "openai_key.txt".
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

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to a Base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {image_path}")

    def get_ood_description(
        self,
        prompt_file: str,
        ood_image_path: str,
        output_file: str = None,
        env_name: str = "",
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate a description for an OOD state using an image and a prompt.

        Args:
            ood_image_path (str): Path to the image representing the OOD state.
            prompt_file (str, optional): Path to the file containing the text prompt template.
                Defaults to None.
            output_file (str, optional): Path to save the generated response. If None, the 
                response is not saved to a file. Defaults to None.
            env_name (str, optional): Name of the environment to substitute in the prompt.
                Defaults to an empty string.
            max_tokens (int, optional): Maximum number of tokens for the API response.
                Defaults to 500.

        Returns:
            Optional[str]: The generated description of the OOD state, or None if an error occurs.

        Raises:
            FileNotFoundError: If the prompt file or image file is not found.
            ValueError: If neither a direct prompt nor a prompt file is provided.
        """
        try:
            # Handle the prompt input
            if prompt_file is None:
                raise ValueError("A prompt file must be provided.")

            # Read the prompt from the file
            prompt = self._load_file(prompt_file)
            
            # Substitute the environment name if provided
            if prompt and env_name:
                prompt = prompt.replace("{env_name}", env_name)

            # Encode the image to a Base64 string
            base64_image = self._encode_image_to_base64(ood_image_path)

            # Prepare the input for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in generating detailed image-to-text descriptions."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
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
            print(f"Error in get_ood_description: {e}")
            return None


def main():
    """
    Demonstrates the usage of the OODDescriptor class for generating descriptions 
    of OOD states using GPT-4 Vision.
    """
    try:
        args = get_args()

        # Initialize the OODDescriptor
        ood_descriptor = OODDescriptor()

        # Generate a description of the OOD state
        response = ood_descriptor.get_ood_description(
            prompt_file="prompts/ood_description.txt",
            ood_image_path=f"ood_images/{args.env}_ood.png",
            output_file=f"outputs/{args.env}/ood_description_output.txt",
            env_name=f"{args.env.capitalize()} environment"
        )

        # Print the response if available
        if response:
            print("OOD Description:", response)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
