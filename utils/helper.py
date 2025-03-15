import os

def create_directories(directories: dict) -> None:
  """
  Creates directories from a dictionary of paths if they don't exist.
  
  Args:
      directories (dict): Dictionary containing directory paths
  """
  for directory in directories.values():
    if not os.path.exists(directory):
      os.makedirs(directory)