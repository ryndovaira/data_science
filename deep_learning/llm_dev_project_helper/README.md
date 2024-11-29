Main goal - Create a intellij (pycharm, webstorm, intellij) plugin that will help developers to get insites about project, review files as fellow developer and provide extencieve checks.
Plugin language - Kotlin (or python if it is possible)
I will use Chatgpt API with the biggest context window.
Framefork - LangChain.

basic flow:
1. User opens the plugin
2. User selects the project root
3. Plugin scans the project and creates a tree of files
4. User selects the files or the whole tree
5. User can select what kind of checks, insites and help he wants (e.g. code review, code quality, code style, write README.md etc.)
6. Plugin sends the files to my application that uses ChatGPT API
7. Plugin gets the response and shows it to the user