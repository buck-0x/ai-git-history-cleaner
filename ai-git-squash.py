#!/usr/bin/env python3
"""
AI Git Squash - A tool to help squash Git commits with AI assistance
"""

import argparse
import os
import sys
import subprocess

try:
    import git
    import openai
except ImportError:
    print("Required packages not installed. Install with:")
    print("pip install gitpython openai")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI-assisted Git commit squashing")
    parser.add_argument('--repo', type=str, default='.', 
                      help='Path to the git repository (defaults to current directory)')
    parser.add_argument('--count', type=int, default=5,
                      help='Number of commits to consider squashing (default: 5)')
    parser.add_argument('--api-key', type=str, 
                      help='OpenAI API key (defaults to OPENAI_API_KEY environment variable)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    
    return parser.parse_args()

def get_commit_messages(repo_path, count):
    """Get the last N commit messages from the repository."""
    try:
        repo = git.Repo(repo_path)
        commits = list(repo.iter_commits('HEAD', max_count=count))
        return [(commit.hexsha[:7], commit.message.strip()) for commit in commits]
    except git.InvalidGitRepositoryError:
        print(f"Error: {repo_path} is not a valid git repository")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing git repository: {e}")
        sys.exit(1)

def generate_squash_message(commits, api_key):
    """Generate a squashed commit message using OpenAI."""
    # Set up OpenAI API
    if api_key:
        openai.api_key = api_key
    elif 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        print("Error: OpenAI API key not provided")
        print("Either use --api-key or set the OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Format the commits for the prompt
    commit_list = "\n".join([f"{sha}: {msg}" for sha, msg in commits])
    
    prompt = f"""
The following are git commit messages. Create a single concise commit message that 
summarizes all the changes effectively. Make it follow good commit message practices.

Commits to squash:
{commit_list}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in writing good git commit messages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating commit message with OpenAI: {e}")
        return None

def perform_squash(repo_path, count, squash_message, dry_run=False):
    """Perform the actual git squash operation."""
    try:
        if dry_run:
            print(f"Would squash the last {count} commits with message:")
            print(f"\n{squash_message}\n")
            return True
        
        # Change to the repo directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Perform the squash using git commands
        # We'll use git reset --soft to keep changes and then recommit
        target_commit = f"HEAD~{count}"
        
        # Get the current branch name
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, check=True)
        current_branch = result.stdout.strip()
        
        print(f"Squashing the last {count} commits on branch '{current_branch}'...")
        
        # Reset to the target commit but keep changes staged
        subprocess.run(['git', 'reset', '--soft', target_commit], check=True)
        
        # Create a new commit with our generated message
        subprocess.run(['git', 'commit', '-m', squash_message], check=True)
        
        print("Squash completed successfully!")
        
        # Return to original directory
        os.chdir(original_dir)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        # Return to original directory in case of error
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"Error during squash operation: {e}")
        # Return to original directory in case of error
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False

def main():
    """Main function."""
    args = parse_args()
    
    print(f"Analyzing the last {args.count} commits...")
    commits = get_commit_messages(args.repo, args.count)
    
    if not commits:
        print("No commits found to squash.")
        return
    
    print("Found the following commits:")
    for sha, msg in commits:
        # Only show first line of each commit message
        first_line = msg.split('\n')[0]
        print(f"  {sha}: {first_line}")
    
    print("\nGenerating squashed commit message...")
    squash_message = generate_squash_message(commits, args.api_key)
    
    if not squash_message:
        print("Failed to generate a squash message. Aborting.")
        return
    
    print(f"\nProposed squash message:\n{squash_message}\n")
    
    if args.dry_run:
        print("Dry run mode - no changes will be made.")
        perform_squash(args.repo, args.count, squash_message, dry_run=True)
        return
    
    confirm = input("Proceed with squash? [y/N] ").lower()
    if confirm in ('y', 'yes'):
        success = perform_squash(args.repo, args.count, squash_message)
        if success:
            print("Squash operation completed.")
    else:
        print("Squash operation cancelled.")

if __name__ == "__main__":
    main()
