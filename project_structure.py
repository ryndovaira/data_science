import os
import argparse


def generate_tree_structure(directory, prefix=""):
    """
    Recursively generates a visual representation of the directory and file hierarchy in a tree-like structure.

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
        if os.path.isfile(entry_path):
            tree.append(f"{prefix}{connector}{entry}")
        elif os.path.isdir(entry_path) and not entry.startswith(
            (".", "__pycache__", ".git", ".idea", ".ipynb_checkpoints")
        ):
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


def main():
    """
    Main function to handle user input and generate the tree structure.
    """
    parser = argparse.ArgumentParser(description="Generate a tree structure for a given directory.")
    parser.add_argument(
        "directory",
        metavar="directory",
        type=str,
        help="The root directory for generating the tree structure.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output_file",
        type=str,
        default="project_structure.md",
        help="The output file where the tree structure will be saved. Default is 'project_structure.md'.",
    )

    args = parser.parse_args()

    root_directory = os.path.abspath(args.directory)
    if not os.path.isdir(root_directory):
        print(f"Error: {root_directory} is not a valid directory.")
        return

    print(f"Scanning directory: {root_directory}")
    root_name = os.path.basename(root_directory.rstrip("/"))
    tree = [
        "```",
        f"{root_name}/",
    ]  # Add Markdown header and start code block
    tree.extend(generate_tree_structure(root_directory))
    tree.append("```")  # Close the code block

    save_tree_to_file(tree, args.output)
    print(f"Tree structure saved to {args.output}")


if __name__ == "__main__":
    main()
