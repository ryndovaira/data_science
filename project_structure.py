import os


def generate_tree_structure(directory, prefix=""):
    """
    Recursively generates a visual representation of the directory and file hierarchy in a tree-like structure.

    The function mimics the style of the 'tree' command output commonly used in Unix/Linux systems:
    - Uses "├──" and "└──" to represent branches in the tree.
    - Employs "│" for vertical connections to indicate sublevels.
    - Indentation is handled with spaces for readability and to align elements at different levels.

    Parameters:
        directory (str): The root directory from which to generate the tree structure.
        prefix (str): A string used to add indentation and connectors for nested elements.

    Returns:
        list[str]: A list of strings where each string represents a line in the tree structure.
                   Each line corresponds to a file or directory with appropriate indentation.
    """
    tree = []
    entries = sorted(os.listdir(directory))
    for index, entry in enumerate(entries):
        entry_path = os.path.join(directory, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        if os.path.isfile(entry_path) and not entry.startswith("."):  # Exclude hidden files
            tree.append(f"{prefix}{connector}{entry}")
        elif os.path.isdir(entry_path) and not entry.startswith("."):  # Exclude hidden directories
            tree.append(f"{prefix}{connector}{entry}/")
            tree.extend(
                generate_tree_structure(
                    entry_path, prefix=prefix + ("    " if index == len(entries) - 1 else "│   ")
                )
            )
    return tree


def save_tree_to_file(tree, output_file):
    """
    Save the generated tree structure to a file.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(tree))


if __name__ == "__main__":
    # Set the project directory and output file
    project_directory = os.getcwd()  # Change to your project's root directory if needed
    output_file = "project_structure.md"

    # Generate the tree structure
    root_name = os.path.basename(project_directory.rstrip("/"))
    print(f"Scanning project directory: {project_directory}")
    tree = [
        "```",
        f"{root_name}/",
    ]  # Add Markdown header and start code block
    tree.extend(generate_tree_structure(project_directory))
    tree.append("```")  # Close the code block

    # Save to file
    save_tree_to_file(tree, output_file)
    print(f"Project structure saved to {output_file}")
