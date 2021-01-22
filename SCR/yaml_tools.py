import yaml
import os_tools


def read_yaml(path, filename):
    """This function will take a path and filename then read the yaml file and return a yaml object

       Args:
           path (str): The path from root to the yaml file to read.
           filename (str): The filename of the yaml to read.

       Returns:
           object: The output of the read file.
    """
    old_path = os_tools.Change_Working_Path(path)
    with open(filename, 'r') as file:
        settings = yaml.safe_load(file)
    os_tools.Change_Working_Path(old_path)
    return settings
