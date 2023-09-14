import os

"""avoid overwriting files"""

def count_folders_by_name(directory, folder_name=None):
    """
    return number of folders with same name in a directory
    directory: pathlib.Path
    """
    return len([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])
