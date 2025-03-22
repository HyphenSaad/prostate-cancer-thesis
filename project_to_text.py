import os

def get_all_python_files(root_dir='.', exclude_file=None):
    """
    Recursively get all Python files from the given directory
    
    Args:
        root_dir (str): Root directory to start the search from
        exclude_file (str): Path of the file to exclude
    
    Returns:
        list: List of paths to Python files
    """
    python_files = []
    
    # Normalize the exclude_file path for comparison
    if exclude_file:
        exclude_file = os.path.normpath(exclude_file)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                # Skip the file if it matches the exclude path
                if exclude_file and os.path.normpath(os.path.abspath(file_path)) == exclude_file:
                    continue
                python_files.append(file_path)
    
    return python_files

def create_consolidated_file(output_file='project_code.txt'):
    """
    Create a consolidated text file containing all Python code
    
    Args:
        output_file (str): Name of the output file
    """
    # Get the absolute path of the current script to exclude it
    current_script = os.path.abspath(__file__)
    python_files = get_all_python_files(exclude_file=current_script)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in python_files:
            # Normalize file path for consistent output
            normalized_path = file_path.replace('\\', '/')
            if normalized_path.startswith('./'):
                normalized_path = normalized_path[2:]
            
            f_out.write(f"[FILE-PATH: {normalized_path}]\n")
            f_out.write("[CODE]\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    code_content = f_in.read()
                    f_out.write(code_content)
            except UnicodeDecodeError:
                # Try different encoding if utf-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as f_in:
                        code_content = f_in.read()
                        f_out.write(code_content)
                except Exception as e:
                    f_out.write(f"# Error reading file: {str(e)}\n")
            
            # Add a newline between files for better readability
            f_out.write("\n\n")
            
    print(f"Consolidated code written to {output_file}")

if __name__ == '__main__':
    create_consolidated_file()
