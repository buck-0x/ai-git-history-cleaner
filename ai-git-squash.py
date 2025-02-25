#!/usr/bin/env python3
"""
AI Git Squash - A tool to help squash Git commits with AI assistance
"""

import argparse
import os
import sys
import subprocess
import time
import json

try:
    import git
    import openai
    from dotenv import load_dotenv
except ImportError:
    print("Required packages not installed. Install with:")
    print("pip install gitpython openai python-dotenv")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI-assisted Git commit squashing")
    parser.add_argument('--repo', type=str, default='.', 
                      help='Path to the git repository (defaults to current directory)')
    parser.add_argument('--count', type=int, default=5,
                      help='Number of commits to consider squashing (default: 5)')
    parser.add_argument('--source-branch', type=str,
                      help='Source branch containing commits to squash (defaults to current branch)')
    parser.add_argument('--target-branch', type=str,
                      help='Target branch for the squashed commit (defaults to source branch)')
    parser.add_argument('--create-target', action='store_true',
                      help='Create the target branch if it does not exist')
    parser.add_argument('--api-key', type=str, 
                      help='OpenAI API key (defaults to OPENAI_API_KEY environment variable)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    parser.add_argument('--logical-grouping', action='store_true',
                      help='Group commits logically based on purpose instead of squashing all into one')
    
    return parser.parse_args()

def get_commit_messages(repo_path, count, source_branch=None, target_branch=None):
    """Get commit messages between branches or the last N commits from a branch."""
    try:
        repo = git.Repo(repo_path)
        
        # Get current branch if source branch not specified
        if not source_branch:
            source_branch = repo.active_branch.name
            print(f"Using current branch '{source_branch}' as source")
        
        # If target branch is specified, get commits between branches
        if target_branch:
            try:
                # Get commits that are in source_branch but not in target_branch
                commits_between = list(repo.iter_commits(f"{target_branch}..{source_branch}"))
                if not commits_between:
                    print(f"No unique commits found in '{source_branch}' compared to '{target_branch}'")
                    sys.exit(0)
                
                # Limit to the specified count if needed
                commits = commits_between[:count] if len(commits_between) > count else commits_between
                print(f"Found {len(commits)} commits in '{source_branch}' not in '{target_branch}'")
            except Exception as e:
                print(f"Error comparing branches: {e}")
                sys.exit(1)
        else:
            # Just get the last N commits from the source branch
            commits = list(repo.iter_commits(source_branch, max_count=count))
            
        return [(commit.hexsha[:7], commit.message.strip()) for commit in commits]
    except git.InvalidGitRepositoryError:
        print(f"Error: {repo_path} is not a valid git repository")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing git repository: {e}")
        sys.exit(1)

def get_logical_commit_groups(commits, api_key):
    """Group commits logically based on purpose and generate squash messages for each group."""
    # Set up OpenAI API
    if api_key:
        openai.api_key = api_key
    elif 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        print("Error: OpenAI API key not provided")
        print("Either use --api-key or set the OPENAI_API_KEY environment variable")
        print("You can also create a .env file with OPENAI_API_KEY=your_api_key")
        sys.exit(1)
    
    # Format the commits for the prompt
    commit_details = "\n".join([f"{sha}: {msg}" for sha, msg in commits])
    
    prompt = f"""
Analyze these git commits and group them logically based on what they're trying to achieve.
Each group should represent a coherent unit of work.

For each group:
1. Generate a concise, meaningful commit message
2. List the commit SHAs that belong in that group

Format your response as JSON like this:
{{
  "groups": [
    {{
      "message": "Add user authentication feature",
      "commits": ["abc1234", "def5678"]
    }},
    {{
      "message": "Fix pagination bugs",
      "commits": ["ghi9012"]
    }}
  ]
}}

Commits to analyze:
{commit_details}
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing git commits and grouping them logically."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        return result["groups"]
    except Exception as e:
        print(f"Error generating logical commit groups with OpenAI: {e}")
        return None

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
        print("You can also create a .env file with OPENAI_API_KEY=your_api_key")
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
            model="gpt-4o",
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

def perform_squash(repo_path, count, squash_message, source_branch=None, target_branch=None, create_target=False, dry_run=False):
    """Perform the actual git squash operation."""
    try:
        # Change to the repo directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Get the repo object
        repo = git.Repo('.')
        
        # Get current branch if source branch not specified
        if not source_branch:
            source_branch = repo.active_branch.name
        
        if dry_run:
            if target_branch:
                print(f"Would squash commits from '{source_branch}' into '{target_branch}' with message:")
            else:
                print(f"Would squash the last {count} commits on branch '{source_branch}' with message:")
            print(f"\n{squash_message}\n")
            return True
        
        # Save the current branch to return to it later if needed
        original_branch = repo.active_branch.name
        
        # If we're working with branches, we need a different approach
        if target_branch:
            print(f"Squashing commits from '{source_branch}' into '{target_branch}'...")
            
            # Check if target branch exists
            target_exists = target_branch in [ref.name for ref in repo.references if isinstance(ref, git.Head)]
            if not target_exists:
                if create_target:
                    print(f"Target branch '{target_branch}' does not exist, creating it...")
                    # Create the target branch from the current HEAD
                    subprocess.run(['git', 'branch', target_branch], check=True)
                else:
                    print(f"Error: Target branch '{target_branch}' does not exist.")
                    print("Use --create-target to create it automatically.")
                    return False
            
            # Create a temporary branch from the target branch
            temp_branch = f"temp-squash-{int(time.time())}"
            subprocess.run(['git', 'checkout', target_branch], check=True)
            subprocess.run(['git', 'checkout', '-b', temp_branch], check=True)
            
            # Cherry-pick commits from source branch in reverse (oldest first)
            # Get the commit range
            result = subprocess.run(['git', 'rev-list', '--reverse', f"{target_branch}..{source_branch}"], 
                                  capture_output=True, text=True, check=True)
            commits = result.stdout.strip().split('\n')
            
            # Filter to respect the count parameter
            if commits and len(commits) > count:
                commits = commits[-count:]
            
            if not commits or commits[0] == '':
                print("No commits to cherry-pick.")
                # Clean up - go back to original branch
                subprocess.run(['git', 'checkout', original_branch], check=True)
                subprocess.run(['git', 'branch', '-D', temp_branch], check=True)
                return False
            
            # Cherry-pick each commit
            for commit in commits:
                try:
                    subprocess.run(['git', 'cherry-pick', '--no-commit', commit], check=True)
                except subprocess.CalledProcessError:
                    print(f"Conflict during cherry-pick of {commit[:7]}. Resolving...")
                    # Stage all files to mark conflicts as resolved
                    subprocess.run(['git', 'add', '.'], check=True)
            
            # Now create a single commit with our squash message
            subprocess.run(['git', 'commit', '-m', squash_message], check=True)
            
            # Switch to target branch and merge the temp branch
            subprocess.run(['git', 'checkout', target_branch], check=True)
            subprocess.run(['git', 'merge', '--ff-only', temp_branch], check=True)
            
            # Delete the temp branch
            subprocess.run(['git', 'branch', '-D', temp_branch], check=True)
            
            # Go back to the original branch if it wasn't the target branch
            if original_branch != target_branch:
                subprocess.run(['git', 'checkout', original_branch], check=True)
        else:
            # Simple case - squash commits on the current branch
            print(f"Squashing the last {count} commits on branch '{source_branch}'...")
            
            # Make sure we're on the right branch
            if original_branch != source_branch:
                subprocess.run(['git', 'checkout', source_branch], check=True)
            
            # Reset to the target commit but keep changes staged
            target_commit = f"HEAD~{count}"
            subprocess.run(['git', 'reset', '--soft', target_commit], check=True)
            
            # Create a new commit with our generated message
            subprocess.run(['git', 'commit', '-m', squash_message], check=True)
            
            # Go back to the original branch if needed
            if original_branch != source_branch:
                subprocess.run(['git', 'checkout', original_branch], check=True)
        
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

def perform_logical_squashes(repo_path, commits, commit_groups, source_branch=None, target_branch=None, create_target=False, dry_run=False):
    """Perform multiple squash operations based on logical commit groups."""
    original_dir = os.getcwd()
    success = True
    
    try:
        # Change to the repo directory
        os.chdir(repo_path)
        repo = git.Repo('.')
        
        # Get current branch if source branch not specified
        if not source_branch:
            source_branch = repo.active_branch.name
            
        # Save the current branch to return to it later
        original_branch = repo.active_branch.name
        
        if dry_run:
            print("Dry run mode - would perform the following squashes:")
            for i, group in enumerate(commit_groups):
                print(f"\nGroup {i+1}: {group['message']}")
                print(f"  Commits: {', '.join(group['commits'])}")
            return True
            
        # In target branch mode, we need to create a temporary branch
        if target_branch:
            # Check if target branch exists
            target_exists = target_branch in [ref.name for ref in repo.references if isinstance(ref, git.Head)]
            if not target_exists:
                if create_target:
                    print(f"Target branch '{target_branch}' does not exist, creating it...")
                    # Create the target branch from the current HEAD
                    subprocess.run(['git', 'branch', target_branch], check=True)
                else:
                    print(f"Error: Target branch '{target_branch}' does not exist.")
                    print("Use --create-target to create it automatically.")
                    return False
                    
            # Create a temporary branch from the target branch
            temp_branch = f"temp-logical-squash-{int(time.time())}"
            subprocess.run(['git', 'checkout', target_branch], check=True)
            subprocess.run(['git', 'checkout', '-b', temp_branch], check=True)
            
            # Process each group
            for i, group in enumerate(commit_groups):
                print(f"\nProcessing group {i+1}/{len(commit_groups)}: {group['message']}")
                
                # Cherry-pick each commit in the group
                for commit_sha in group['commits']:
                    try:
                        # Find the full SHA from the short SHA
                        matching_commits = [c for sha, c in commits if sha.startswith(commit_sha)]
                        if not matching_commits:
                            print(f"Warning: Could not find commit {commit_sha}, skipping")
                            continue
                            
                        full_sha = repo.git.rev_parse(commit_sha)
                        subprocess.run(['git', 'cherry-pick', '--no-commit', full_sha], check=True)
                    except subprocess.CalledProcessError:
                        print(f"Conflict during cherry-pick of {commit_sha}. Resolving...")
                        # Stage all files to mark conflicts as resolved
                        subprocess.run(['git', 'add', '.'], check=True)
                
                # Create a commit for this group
                subprocess.run(['git', 'commit', '-m', group['message']], check=True)
            
            # Merge the temp branch into the target branch
            subprocess.run(['git', 'checkout', target_branch], check=True)
            subprocess.run(['git', 'merge', '--ff-only', temp_branch], check=True)
            
            # Delete the temp branch
            subprocess.run(['git', 'branch', '-D', temp_branch], check=True)
            
            # Go back to the original branch
            if original_branch != target_branch:
                subprocess.run(['git', 'checkout', original_branch], check=True)
                
        else:
            # For source branch only, process each group separately
            # This is trickier as we need to use rebase interactive
            
            print("Creating a temporary script for interactive rebase...")
            # Create a temporary script for git rebase
            rebase_script = "#!/bin/sh\n"
            
            # Map original SHAs to their position in the rebase
            sha_to_line = {}
            all_shas = []
            
            # Get the commits in reverse order (oldest first)
            for i, (sha, _) in enumerate(reversed(commits)):
                sha_to_line[sha] = i + 1
                all_shas.append(sha)
            
            # For each group
            for group in commit_groups:
                # Find the first and last commit in the group
                group_shas = []
                for short_sha in group['commits']:
                    matching_shas = [sha for sha in all_shas if sha.startswith(short_sha)]
                    group_shas.extend(matching_shas)
                
                if not group_shas:
                    continue
                    
                # Pick the first commit to keep
                first_sha = min(sha_to_line[sha] for sha in group_shas)
                
                # Mark remaining commits in group for squashing
                for sha in group_shas:
                    line_num = sha_to_line[sha]
                    if line_num == first_sha:
                        rebase_script += f"sed -i '{line_num}s/^pick/pick/' $1\n"
                    else:
                        rebase_script += f"sed -i '{line_num}s/^pick/squash/' $1\n"
                
                # Set the commit message for this group
                rebase_script += f"echo '{group['message']}' > .git/COMMIT_EDITMSG\n"
            
            # Write the script to a temporary file
            with open('/tmp/rebase-script.sh', 'w') as f:
                f.write(rebase_script)
            os.chmod('/tmp/rebase-script.sh', 0o755)
            
            # Make sure we're on the right branch
            if original_branch != source_branch:
                subprocess.run(['git', 'checkout', source_branch], check=True)
            
            # Start the interactive rebase with our script
            target_commit = f"HEAD~{len(commits)}"
            os.environ['GIT_SEQUENCE_EDITOR'] = '/tmp/rebase-script.sh'
            
            try:
                subprocess.run(['git', 'rebase', '-i', target_commit], check=True)
                print("Logical squash completed successfully!")
            except subprocess.CalledProcessError:
                print("Error during rebase. You may need to manually complete the rebase.")
                success = False
            
            # Clean up
            if os.path.exists('/tmp/rebase-script.sh'):
                os.unlink('/tmp/rebase-script.sh')
            
            # Go back to the original branch if needed
            if original_branch != source_branch and original_branch != repo.active_branch.name:
                subprocess.run(['git', 'checkout', original_branch], check=True)
        
        # Return to original directory
        os.chdir(original_dir)
        return success
        
    except Exception as e:
        print(f"Error during logical squash operation: {e}")
        # Return to original directory in case of error
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Determine branch context for display message
    if args.target_branch:
        print(f"Analyzing commits from '{args.source_branch or 'current branch'}' to '{args.target_branch}'...")
    else:
        print(f"Analyzing the last {args.count} commits from '{args.source_branch or 'current branch'}'...")
    
    commits = get_commit_messages(args.repo, args.count, args.source_branch, args.target_branch)
    
    if not commits:
        print("No commits found to squash.")
        return
    
    print("Found the following commits:")
    for sha, msg in commits:
        # Only show first line of each commit message
        first_line = msg.split('\n')[0]
        print(f"  {sha}: {first_line}")
    
    # Handle logical grouping mode
    if args.logical_grouping:
        print("\nAnalyzing commits to create logical groups...")
        commit_groups = get_logical_commit_groups(commits, args.api_key)
        
        if not commit_groups:
            print("Failed to create logical commit groups. Aborting.")
            return
            
        print("\nProposed logical commit groups:")
        for i, group in enumerate(commit_groups):
            print(f"\nGroup {i+1}: {group['message']}")
            print("  Commits:")
            for commit_sha in group['commits']:
                # Find the full commit details
                matching = [c for sha, c in commits if sha.startswith(commit_sha)]
                if matching:
                    first_line = matching[0].split('\n')[0]
                    matching_sha = next((sha for sha, _ in commits if sha.startswith(commit_sha)), commit_sha)
                    print(f"    {matching_sha}: {first_line}")
        
        if args.dry_run:
            print("\nDry run mode - no changes will be made.")
            perform_logical_squashes(args.repo, commits, commit_groups, 
                                   args.source_branch, args.target_branch, args.create_target, dry_run=True)
            return
        
        confirm = input("\nProceed with logical squashes? [y/N] ").lower()
        if confirm in ('y', 'yes'):
            success = perform_logical_squashes(args.repo, commits, commit_groups,
                                             args.source_branch, args.target_branch, args.create_target)
            if success:
                print("Logical squash operation completed.")
            else:
                print("Logical squash operation had errors.")
        else:
            print("Logical squash operation cancelled.")
    
    # Original single-squash mode
    else:
        print("\nGenerating squashed commit message...")
        squash_message = generate_squash_message(commits, args.api_key)
        
        if not squash_message:
            print("Failed to generate a squash message. Aborting.")
            return
        
        print(f"\nProposed squash message:\n{squash_message}\n")
        
        if args.dry_run:
            print("Dry run mode - no changes will be made.")
            perform_squash(args.repo, args.count, squash_message, 
                         args.source_branch, args.target_branch, args.create_target, dry_run=True)
            return
        
        confirm = input("Proceed with squash? [y/N] ").lower()
        if confirm in ('y', 'yes'):
            success = perform_squash(args.repo, args.count, squash_message, 
                                   args.source_branch, args.target_branch, args.create_target)
            if success:
                print("Squash operation completed.")
        else:
            print("Squash operation cancelled.")

if __name__ == "__main__":
    main()
