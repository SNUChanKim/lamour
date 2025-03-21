import sys
from pathlib import Path
from openai import OpenAI
from typing import Optional
import argparse

sys.path.append(str(Path(__file__).parent))
from ood_description import OODDescriptor
from behavior_reasoning import BehaviorReasoner
from code_generator import CodeGenerator

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

class LaMoUR:
    """
    A class to integrate and manage the functionality of OOD description generation, 
    behavior reasoning, and Python code generation for reinforcement learning tasks.

    Attributes:
        api_key (str): The API key for accessing the OpenAI API.
        client (OpenAI): The OpenAI API client instance.
        ood_descriptor (OODDescriptor): Handles out-of-distribution state description generation.
        behavior_reasoner (BehaviorReasoner): Handles reasoning for recovery behaviors.
        code_generator (CodeGenerator): Handles Python code generation based on prompts.
    """

    def __init__(self, api_key_path: Optional[str] = None):
        """
        Initialize the LaMoUR class with OpenAI API client and supporting modules.

        Args:
            api_key_path (str, optional): Path to the file containing the OpenAI API key. 
                Defaults to "openai_key.txt".
        """
        if api_key_path is None:
            api_key_path = Path(__file__).parent / "openai_key.txt"
            
        self.api_key = self._load_api_key(api_key_path)
        self.client = OpenAI(api_key=self.api_key)
        
        self.ood_descriptor = OODDescriptor(self.client)
        self.behavior_reasoner = BehaviorReasoner(self.client)
        self.code_generator = CodeGenerator(self.client)
        
        self.ood_description_prompt = Path(__file__).parent / "prompts/ood_description.txt" 
        self.behavior_reasoning_prompt = Path(__file__).parent / "prompts/behavior_reasoning.txt"
        self.code_generation_prompt = Path(__file__).parent / "prompts/code_generation.txt"
        self.few_shot_prompt = Path(__file__).parent / "prompts/example.txt"

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
        
    def generate_codes(
        self, 
        ood_image_path: str, 
        original_task: str,
        environment_description_file: str,
        ood_description_output_file: str, 
        behavior_reasoning_output_file: str, 
        code_generation_output_file: str,
        env_name: str,
        use_few_shot: bool = None
    ):
        """
        Generate OOD descriptions, recovery reasoning, and Python code based on the provided input prompts.

        Args:
            ood_image_path (str): Path to the image file representing the agent's OOD state.
            original_task (str): A description of the agent's original task.
            environment_description_file (str): Path to the file describing the agent's environment.
            ood_description_output_file (str): Path to save the generated OOD state description.
            behavior_reasoning_output_file (str): Path to save the generated reasoning for recovery behavior.
            code_generation_output_file (str): Path to save the generated Python code.
            env_name (str): The name of the environment in which the agent operates.
            
        Returns:
            str: The generated Python code for the agent's recovery behavior.
        """
        ood_description = self.ood_descriptor.get_ood_description(
            prompt_file=self.ood_description_prompt, 
            ood_image_path=ood_image_path, 
            output_file=ood_description_output_file, 
            env_name=env_name
        )
        
        behavior_reasoning = self.behavior_reasoner.get_reasoning(
            prompt_file=self.behavior_reasoning_prompt, 
            ood_description=ood_description, 
            output_file=behavior_reasoning_output_file, 
            original_task=original_task, 
            env_name=env_name
        )
        
        few_shot_example = None
        if use_few_shot:
            few_shot_example = self.few_shot_prompt
        
        response_code = self.code_generator.get_generated_code(
            prompt_file=self.code_generation_prompt, 
            environment_description_file=environment_description_file,
            behavior_reasoning=behavior_reasoning, 
            output_file=code_generation_output_file, 
            env_name=env_name,
            few_shot_example=few_shot_example
        )
        return response_code


def main():
    """
    Demonstrates the usage of the LaMoUR class for generating Python code
    from given input files.
    """
    try:
        args = get_args()
        if args.env == 'pushchair':
            original_task = get_original_task('PushChairNormal-v1')
            use_few_shot = True
        else:
            original_task = get_original_task('AntNormal-v3')
            use_few_shot = False
            
        # Initialize the LaMoUR with your OpenAI API key
        lamour = LaMoUR()

        # Generate Python code directly from input files
        generated_code = lamour.generate_codes(
            ood_image_path=f"ood_images/{args.env}_ood.png",
            original_task=original_task,
            environment_description_file=f"env_description/{args.env}_env.txt",
            ood_description_output_file=f"outputs/{args.env}/ood_description_output.txt", 
            behavior_reasoning_output_file=f"outputs/{args.env}/behavior_reasoning_output.txt", 
            code_generation_output_file=f"outputs/{args.env}/generated_code.py",
            env_name=f"{args.env.capitalize()} environment",
            use_few_shot=use_few_shot
            )

        print(f"Code generation complete. Output saved to outputs/{args.env}/generated_code.py.")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()