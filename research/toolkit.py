import os

def join_paths(*args):
    """
    Join multiple path components into a single path using os.path.join.

    Args:
        *args: Variable number of path components to be joined.

    Returns:
        str: Joined path.
    """
    return os.path.join(*args)

def list_directory_contents(directory_path):
    """
    List the contents (files and directories) of a given directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: List of filenames and directory names in the specified directory.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        return os.listdir(directory_path)
    else:
        print(f"The specified directory '{directory_path}' does not exist or is not a directory.")
        return []
    
def check_path_existence(path):
    """
    Check if a file or directory exists at the specified path.

    Args:
        path (str): The path to the file or directory.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    return os.path.exists(path)

def create_directory(directory_path):
    """
    Create a directory at the specified path, including parent directories if needed.

    Args:
        directory_path (str): The path to the directory to be created.

    Returns:
        None
    """
    os.makedirs(directory_path, exist_ok=True)



