# Annotate Stimulus

This module provides functionality to annotate stimuli with their corresponding trigger types using the OpenAI API. The `annotate_stimulus` function has been optimized to run in parallel using Python's `multiprocessing` module, allowing for faster processing of large datasets.

## Functionality

- **annotate_stimulus**: Annotates a list of stimuli with their trigger types based on a specified emotion. Utilizes multiprocessing to enhance performance.

## Usage

1. Ensure the OpenAI API key and base URL are set in the environment variables.
2. Prepare the input data in JSON format.
3. Call the `annotate_stimulus` function with the appropriate parameters.
