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
    parser.add_argument('--count', type=int, default=0,
                      help='Number of commits to consider squashing (default: all since branch creation)')
    parser.add_argument('--source-branch', type=str,
                      help='Source branch containing commits to squash (defaults to current branch)')
    parser.add_argument('--target-branch', type=str,
                      help='Target branch for the squashed commit (defaults to source-branch-squashed)')
    parser.add_argument('--create-target', action='store_true', default=True,
                      help='Create the target branch if it does not exist (default: True)')
    parser.add_argument('--new-root', action='store_true', default=False,
                      help='Create a new root branch without shared history (useful for replacing main)')
    parser.add_argument('--api-key', type=str, 
                      help='OpenAI API key (defaults to OPENAI_API_KEY environment variable)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    parser.add_argument('--logical-grouping', action='store_true', default=True,
                      help='Group commits logically based on purpose (default: True)')
    
    return parser.parse_args()

def get_commit_messages(repo_path, count, source_branch=None, target_branch=None):
    """Get commit messages from a branch."""
    try:
        repo = git.Repo(repo_path)
        
        # Get current branch if source branch not specified
        if not source_branch:
            source_branch = repo.active_branch.name
            print(f"Using current branch '{source_branch}' as source")
        
        # For the common use case, we want to get ALL commits from the source branch
        # that would be applied to a new branch
        try:
            # Get the merge-base (common ancestor) with main/master if target_branch not specified
            base_branch = "main"
            if not repo.heads.get(base_branch):
                base_branch = "master"
                if not repo.heads.get(base_branch):
                    # If neither main nor master exists, just use the last N commits
                    print(f"No main or master branch found, using last {count} commits from '{source_branch}'")
                    commits = list(repo.iter_commits(source_branch, max_count=count))
                    return [(commit.hexsha[:7], commit.message.strip()) for commit in commits]
            
            # Get the common ancestor
            merge_base = repo.git.merge_base(source_branch, base_branch)
            
            # Get all commits from merge_base to source_branch head
            commits = list(repo.iter_commits(f"{merge_base}..{source_branch}"))
            
            # If there are too many, limit to count
            if len(commits) > count and count > 0:
                print(f"Found {len(commits)} commits, limiting to {count} as specified")
                commits = commits[:count]
            else:
                print(f"Found {len(commits)} commits in '{source_branch}' since branching from {base_branch}")
        except Exception as e:
            print(f"Error finding branch history: {e}")
            print(f"Falling back to last {count} commits from '{source_branch}'")
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

def perform_logical_squashes(repo_path, commits, commit_groups, source_branch=None, target_branch=None, create_target=False, new_root=False, dry_run=False):
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
        
        # Find base point for branches
        try:
            if new_root:
                # Create a new orphan branch with no history
                print("Creating a new root branch with no shared history")
                merge_base = None
            else:
                # First try to find the merge-base with main/master
                base_branch = "main"
                if not repo.heads.get(base_branch):
                    base_branch = "master"
                    if not repo.heads.get(base_branch):
                        # If no main/master, just use the parent of the oldest commit we have
                        oldest_commit = commits[-1][0]
                        merge_base = repo.git.rev_parse(f"{oldest_commit}^")
                        print(f"Using parent of oldest commit as base point: {merge_base[:7]}")
                    else:
                        merge_base = repo.git.merge_base(source_branch, base_branch)
                        print(f"Using merge-base with {base_branch} as starting point: {merge_base[:7]}")
                else:
                    merge_base = repo.git.merge_base(source_branch, base_branch)
                    print(f"Using merge-base with {base_branch} as starting point: {merge_base[:7]}")
        except Exception as e:
            print(f"Error finding merge base: {e}")
            if commits:
                oldest_commit = commits[-1][0]
                merge_base = repo.git.rev_parse(f"{oldest_commit}^")
                print(f"Using parent of oldest commit as base point: {merge_base[:7]}")
            else:
                print("No commits found to determine base point.")
                return False
            
        # Check/create target branch
        if target_branch:
            target_exists = target_branch in [ref.name for ref in repo.references if isinstance(ref, git.Head)]
            if not target_exists:
                if create_target:
                    if new_root:
                        print(f"Creating new orphan branch '{target_branch}' with no history...")
                        # Create an orphan branch with no history
                        subprocess.run(['git', 'checkout', '--orphan', target_branch], check=True)
                        # Clean the working directory
                        subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
                    else:
                        print(f"Target branch '{target_branch}' does not exist, creating it at {merge_base[:7]}...")
                        # Create target branch at the merge-base point
                        subprocess.run(['git', 'checkout', '-b', target_branch, merge_base], check=True)
                else:
                    print(f"Error: Target branch '{target_branch}' does not exist.")
                    print("Use --create-target to create it automatically.")
                    return False
            else:
                # Use existing target branch
                print(f"Using existing target branch: {target_branch}")
                subprocess.run(['git', 'checkout', target_branch], check=True)
                    
                if new_root:
                    # For new root, we want to remove all files and history
                    subprocess.run(['git', 'checkout', '--orphan', f'temp-orphan-{int(time.time())}'], check=True)
                    subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
                    # Now recreate the branch
                    subprocess.run(['git', 'branch', '-D', target_branch], check=True)
                    subprocess.run(['git', 'checkout', '-b', target_branch], check=True)
                elif not dry_run and merge_base:
                    # Ask if user wants to reset target branch to merge-base
                    reset_choice = input(f"Reset '{target_branch}' to common ancestor at {merge_base[:7]}? [y/N] ").lower()
                    if reset_choice in ('y', 'yes'):
                        print(f"Resetting '{target_branch}' to {merge_base[:7]}")
                        subprocess.run(['git', 'reset', '--hard', merge_base], check=True)
        else:
            # If no target branch specified, create a new one with a default name
            target_branch = f"{source_branch}-squashed"
            print(f"No target branch specified, creating '{target_branch}'")
            
            # Check if it already exists
            if target_branch in [ref.name for ref in repo.references if isinstance(ref, git.Head)]:
                target_branch = f"{source_branch}-squashed-{int(time.time())}"
                print(f"Branch already exists, using '{target_branch}' instead")
            
            # Create the new branch
            if new_root:
                print(f"Creating new orphan branch '{target_branch}' with no history...")
                # Create an orphan branch with no history
                subprocess.run(['git', 'checkout', '--orphan', target_branch], check=True)
                # Clean the working directory
                subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
            else:
                print(f"Creating '{target_branch}' at {merge_base[:7]}")
                subprocess.run(['git', 'checkout', '-b', target_branch, merge_base], check=True)
        
        # Now we're on the target branch, apply the logical groups
        for i, group in enumerate(commit_groups):
            print(f"\nProcessing group {i+1}/{len(commit_groups)}: {group['message']}")
            
            # Track if we've made any changes in this group
            group_has_changes = False
            
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
                    group_has_changes = True
                except subprocess.CalledProcessError:
                    print(f"Conflict during cherry-pick of {commit_sha}. Resolving...")
                    # Stage all files to mark conflicts as resolved
                    subprocess.run(['git', 'add', '.'], check=True)
                    group_has_changes = True
            
            # Create a commit for this group if we made changes
            if group_has_changes:
                try:
                    subprocess.run(['git', 'commit', '-m', group['message']], check=True)
                    print(f"Created commit for group: {group['message']}")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not create commit for group. Error: {e}")
                    # This could happen if there were no actual changes
                    print("Continuing to next group...")
            else:
                print("No changes in this group, skipping commit")
        
        # Go back to the original branch when done
        if original_branch != target_branch:
            subprocess.run(['git', 'checkout', original_branch], check=True)
            
        # Print instructions for making this the new main branch
        if source_branch == "main" or source_branch == "master":
            print("\n=== INSTRUCTIONS FOR REPLACING MAIN BRANCH ===")
            print(f"Your new clean branch '{target_branch}' is ready.")
            print("To use it as your new main branch, you can:")
            print(f"1. git checkout {target_branch}")
            print(f"2. git branch -m main main-old  # Rename old main")
            print(f"3. git branch -m {target_branch} main  # Make your new branch the main branch")
            print("4. git push -f origin main  # Force push the new main (USE WITH CAUTION!)")
            print("=== END INSTRUCTIONS ===\n")
        
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
    if args.source_branch:
        print(f"Analyzing commits from branch '{args.source_branch}'...")
    else:
        print(f"Analyzing commits from current branch...")
    
    if args.new_root:
        print("Creating a new branch with no shared history (--new-root mode)")
    
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
                                   args.source_branch, args.target_branch, 
                                   args.create_target, args.new_root, dry_run=True)
            return
        
        confirm = input("\nProceed with logical squashes? [y/N] ").lower()
        if confirm in ('y', 'yes'):
            success = perform_logical_squashes(args.repo, commits, commit_groups,
                                             args.source_branch, args.target_branch, 
                                             args.create_target, args.new_root)
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
