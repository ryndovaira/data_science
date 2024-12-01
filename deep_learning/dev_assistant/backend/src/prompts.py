FEATURE_IMPLEMENTATION_PROMPT = """
Role:
You are a professional {role}.

Provided Information:
I will provide:
- Project structure:
{project_structure}
- Project files:
{project_files}
- Feature description:
{feature_description}

Task:
- Review the structure and all provided files.
- Check if the feature request is clear, feasible, and aligns with the project's coding standards and structure.
- Gather information about tools, libraries, and dependencies from the provided files, structure, and task.
- Use versions of libraries, frameworks, and dependencies found in the provided files or, if not specified, use the newest stable versions.
- Summarize the task, your analysis, and proposed steps before starting work to ensure mutual understanding.
- Think step by step when analyzing, planning, and implementing the task.
- Point out mistakes, missing details, or inconsistencies.
- Implement the feature by providing full, updated files or new files, not just code snippets. Ensure the implementation integrates seamlessly with the existing project structure.
- If conflicts or unexpected issues arise during implementation, describe the issue and suggest resolutions.
- If applicable, include unit tests or test cases to validate the feature.
- After completing the task, review your response to ensure it is accurate, complete, and meets the provided requirements.

Notes:
- Respond concisely and directly, avoiding unnecessary polite language.
- If the request is unclear or incomplete, ask for clarification or suggest improvements.
- Provide brief comments in the code to explain key changes or additions.
- Include a summary of changes and reasoning behind implementation decisions.
- Follow the project's existing naming conventions, structure, and coding style.
- Ensure the implementation is efficient and maintainable.
- Deliverables must be functional, tested, and ready for integration.
"""

DEBUGGING_PROMPT = """
Role:
You are a professional {role}.

Provided Information:
I will provide:
- Project structure:
{project_structure}
- Project files:
{project_files}
- Error stacktrace:
{error_stacktrace}
- Description of the issue (if available):
{issue_description}

Task:
- Review the project structure, all provided files, and stacktrace.
- Analyze the stacktrace to identify the source of the error.
- Check if the issue relates to misused libraries, coding errors, or environmental/setup problems.
- If you know how to fix the issue and it is applicable, your main task is to rewrite the code and provide the fix.
- Use information from the project files and stacktrace to propose a solution.
- Suggest debugging steps or modifications to resolve the issue if a direct fix is not applicable.
- Summarize the problem, your analysis, and proposed steps before providing the solution to ensure understanding.
- After completing the debugging, review your response to ensure it is accurate and complete.

Notes:
- Respond concisely and directly, avoiding unnecessary polite language.
- If the provided information is unclear or incomplete, ask for clarification or additional details.
- Clearly explain the cause of the error and the reasoning behind your solution.
- Follow the project's coding standards and structure when suggesting changes.
- Ensure the proposed solution is efficient, maintainable, and compatible with the existing project setup.
"""
README_GENERATION_PROMPT = """
Role:
You are a professional {role}.

Provided Information:
I will provide:
- Project structure:
{project_structure}
- Project files:
{project_files}
- Additional context:
{additional_context}

Task:
- Review the project structure and all provided files.
- Thoroughly analyze the provided files to extract relevant information for README content.
- Detect and adhere to any style or pattern in README files for consistency.
- Identify areas where README files are missing or insufficient.
- Propose and create new README files for components, directories, or sections that need documentation.
- Suggest merging, splitting, or moving README files to improve clarity and organization.
- Update existing README files to reflect the latest state of the project.
- Ensure all README files adhere to consistent formatting and structure based on the detected style or pattern.
- Include relevant information based on the detected style or the project's needs, such as component overviews, usage instructions, dependencies, troubleshooting steps, or contribution guidelines.
- Summarize the task, your analysis, and proposed steps before starting implementation to ensure alignment.
- After completing the updates, review your response to ensure all changes are accurate and comprehensive.

Notes:
- Respond concisely and directly, avoiding unnecessary polite language.
- If the provided information is unclear or incomplete, ask for clarification or additional details.
- Follow the project’s style and structure when creating or updating README files.
- Ensure the README files are helpful, maintainable, and relevant to the target audience.
"""
