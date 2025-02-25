# AI Git Squash

A tool to help squash Git commits with AI assistance, providing intelligent commit message generation and logical commit grouping.

## Features

- **AI-Powered Commit Message Generation**: Automatically generates meaningful commit messages using OpenAI.
- **Logical Commit Grouping**: Groups consecutive commits based on their purpose.
- **Branch Management**: Create clean branches with squashed commits.
- **New Root Creation**: Option to create completely new branches without shared history.
- **Content Verification**: Ensures that source and target branches have the same content after squashing.
- **Conflict Resolution**: Automatically resolves conflicts by choosing the newer version.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-git-squash.git
   cd ai-git-squash
   ```

2. Run the helper script which will set up a virtual environment and install dependencies:

   ```bash
   chmod +x run-ai-git-squash.sh
   ./run-ai-git-squash.sh --help
   ```

3. Set your OpenAI API key in a `.env` file:

   ```bash
   cp .env.template .env
   # Edit .env with your API key
   ```

## Usage

### Basic Usage

To squash commits in the current branch:
```bash
./run-ai-git-squash.sh
```

### Create a Clean Version of a Branch

```bash
./run-ai-git-squash.sh --source-branch messy-branch --target-branch clean-branch
```

### Replace a Main Branch with a Clean Version

```bash
./run-ai-git-squash.sh --source-branch main --target-branch clean-main --new-root
```

### All Options

```text
--repo PATH             Path to the Git repository (defaults to current directory)
--count NUMBER          Number of commits to consider (default: all since branch creation)
--source-branch NAME    Source branch containing commits to squash
--target-branch NAME    Target branch for the squashed commits
--create-target         Create the target branch if it doesn't exist (default: True)
--new-root              Create a new root branch without shared history
--api-key KEY           OpenAI API key (can also use .env file or environment variable)
--dry-run               Show what would be done without making changes
--logical-grouping      Group consecutive commits logically (default: True)
--batch-size NUMBER     Maximum commits to process in a single API call (default: 100)
--verify                Verify branch content equality after squashing (default: True)
--skip-verify           Skip verification of branch content
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
