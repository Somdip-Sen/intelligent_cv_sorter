def load_prompt(file_path: str) -> str:
    """Loads a prompt from a text file."""
    with open(file_path, 'r') as f:
        return f.read()