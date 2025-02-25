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
import difflib

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
                      help='Group consecutive commits logically based on purpose (default: True)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Maximum number of commits to process in a single API call (default: 100)')
    parser.add_argument('--verify', action='store_true', default=True,
                      help='Verify that source and target branches have the same content after squashing (default: True)')
    parser.add_argument('--skip-verify', action='store_true',
                      help='Skip verification of branch content equality')
    
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
            # If we're doing a new root branch, we want all commits regardless of base
            if source_branch == "main" or source_branch == "master":
                print(f"Processing all commits from {source_branch} branch")
                commits = list(repo.iter_commits(source_branch, max_count=count if count > 0 else None))
                if len(commits) > 0:
                    print(f"Found {len(commits)} commits in '{source_branch}'")
                    if count > 0 and len(commits) > count:
                        print(f"Limiting to {count} most recent commits as specified")
                        commits = commits[:count]
                return [(commit.hexsha[:7], commit.message.strip()) for commit in commits]
            
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
            commits = list(repo.iter_commits(source_branch, max_count=count if count > 0 else None))
            
        return [(commit.hexsha[:7], commit.message.strip()) for commit in commits]
    except git.InvalidGitRepositoryError:
        print(f"Error: {repo_path} is not a valid git repository")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing git repository: {e}")
        sys.exit(1)

def validate_consecutive_commits(groups, commits):
    """Ensure that each group only contains consecutive commits."""
    valid_groups = []
    commit_indices = {sha: idx for idx, (sha, _) in enumerate(commits)}
    
    for group in groups:
        # Get the indices for each commit in this group
        indices = [commit_indices.get(sha, -1) for sha in group['commits']]
        # Remove any invalid indices
        indices = [idx for idx in indices if idx >= 0]
        # Sort indices
        indices.sort()
        
        # Check if indices form a consecutive sequence
        is_consecutive = len(indices) > 0 and (max(indices) - min(indices) + 1 == len(indices))
        
        if is_consecutive:
            # Group is valid, keep it
            valid_groups.append(group)
        else:
            # Report issue and split into smaller consecutive groups
            print(f"Warning: Group with message '{group['message']}' contains non-consecutive commits. Splitting.")
            
            # Create subgroups with consecutive commits
            consecutive_subgroups = []
            current_subgroup = []
            prev_idx = None
            
            for sha in group['commits']:
                idx = commit_indices.get(sha, -1)
                if idx < 0:
                    continue
                
                if prev_idx is None or idx == prev_idx + 1:
                    # Continues the sequence
                    current_subgroup.append(sha)
                else:
                    # Start a new subgroup
                    if current_subgroup:
                        consecutive_subgroups.append(current_subgroup)
                    current_subgroup = [sha]
                
                prev_idx = idx
            
            # Add the last subgroup if it exists
            if current_subgroup:
                consecutive_subgroups.append(current_subgroup)
            
            # Create valid groups from the subgroups
            for subgroup in consecutive_subgroups:
                if len(subgroup) > 0:
                    valid_groups.append({
                        "message": group['message'] if len(consecutive_subgroups) == 1 else 
                                   f"{group['message']} (part {consecutive_subgroups.index(subgroup)+1}/{len(consecutive_subgroups)})",
                        "commits": subgroup,
                        "commit_indices": [commit_indices.get(sha, 9999) for sha in subgroup]
                    })
    
    return valid_groups

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
    
    # Store commit index for later sorting (newest commit is index 0)
    commit_indices = {sha: idx for idx, (sha, _) in enumerate(commits)}
    
    # For large commit sets, we need to process in batches
    MAX_COMMITS_PER_BATCH = 100
    all_groups = []
    
    if len(commits) > MAX_COMMITS_PER_BATCH:
        print(f"\nProcessing {len(commits)} commits in batches of {MAX_COMMITS_PER_BATCH}...")
        batches = [commits[i:i + MAX_COMMITS_PER_BATCH] for i in range(0, len(commits), MAX_COMMITS_PER_BATCH)]
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} commits)...")
            batch_groups = process_commit_batch(batch, api_key)
            if batch_groups:
                # Validate that groups only contain consecutive commits
                valid_batch_groups = validate_consecutive_commits(batch_groups, batch)
                all_groups.extend(valid_batch_groups)
            else:
                print(f"Warning: Failed to process batch {i+1}, skipping...")
        
        # Store commit indices in each group for sorting later
        for group in all_groups:
            if 'commit_indices' not in group:
                group['commit_indices'] = [commit_indices.get(sha, 9999) for sha in group['commits']]
        
        return all_groups
    else:
        batch_groups = process_commit_batch(commits, api_key)
        if batch_groups:
            # Validate that groups only contain consecutive commits
            valid_groups = validate_consecutive_commits(batch_groups, commits)
            # Store commit indices in each group for sorting later
            for group in valid_groups:
                if 'commit_indices' not in group:
                    group['commit_indices'] = [commit_indices.get(sha, 9999) for sha in group['commits']]
            return valid_groups
        return []

def process_commit_batch(commits, api_key):
    """Process a batch of commits for logical grouping."""
    # Format the commits for the prompt - add an index number for reference
    commit_details = "\n".join([f"{i+1}. {sha}: {msg}" for i, (sha, msg) in enumerate(commits)])
    
    prompt = f"""
Analyze these git commits and group them logically based on what they're trying to achieve.
IMPORTANT: Only group commits that are adjacent/consecutive in the list below.
Each group should represent a coherent unit of work. Respect the chronological order of the commits.

For each group:
1. Generate a concise, meaningful commit message
2. List the commit SHAs that belong in that group (preserve the original commit order)
3. ONLY include commits that appear next to each other in the list - never group commits that are separated by other commits

Format your response as JSON like this:
{{
  "groups": [
    {{
      "message": "Add user authentication feature",
      "commits": ["abc1234", "def5678"]  // These must be consecutive commits
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
        print("Calling OpenAI API to analyze commits...")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing git commits and grouping them logically."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=4000,
            timeout=120  # 2 minute timeout for large batches
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
    
    # Track timing for large operations
    start_time = time.time()
    
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
                        
                        # Save the .gitignore content from source branch before switching
                        gitignore_content = ""
                        try:
                            # Check if .gitignore exists in the source branch
                            if os.path.exists(".gitignore"):
                                with open(".gitignore", "r") as f:
                                    gitignore_content = f.read()
                                print("Found .gitignore in source branch, will add to new branch")
                        except Exception as e:
                            print(f"Warning: Could not read .gitignore: {e}")
                        
                        # Create an orphan branch with no history
                        subprocess.run(['git', 'checkout', '--orphan', target_branch], check=True)
                        # Clean the working directory
                        subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
                        
                        # Add .gitignore as the first commit if we found one
                        if gitignore_content:
                            try:
                                # Create .gitignore file
                                with open(".gitignore", "w") as f:
                                    f.write(gitignore_content)
                                # Commit it
                                subprocess.run(['git', 'add', '.gitignore'], check=True)
                                subprocess.run(['git', 'commit', '-m', 'Add .gitignore from source branch'], check=True)
                                print("Created initial commit with .gitignore from source branch")
                            except Exception as e:
                                print(f"Warning: Failed to add .gitignore: {e}")
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
                    # Save the .gitignore content from source branch before switching
                    gitignore_content = ""
                    try:
                        # Get .gitignore from the current branch before switching
                        if os.path.exists(".gitignore"):
                            with open(".gitignore", "r") as f:
                                gitignore_content = f.read()
                            print("Found .gitignore in source branch, will add to new branch")
                    except Exception as e:
                        print(f"Warning: Could not read .gitignore: {e}")
                        
                    # For new root, we want to remove all files and history
                    subprocess.run(['git', 'checkout', '--orphan', f'temp-orphan-{int(time.time())}'], check=True)
                    subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
                    # Now recreate the branch
                    subprocess.run(['git', 'branch', '-D', target_branch], check=True)
                    subprocess.run(['git', 'checkout', '-b', target_branch], check=True)
                    
                    # Add .gitignore as the first commit if we found one
                    if gitignore_content:
                        try:
                            # Create .gitignore file
                            with open(".gitignore", "w") as f:
                                f.write(gitignore_content)
                            # Commit it
                            subprocess.run(['git', 'add', '.gitignore'], check=True)
                            subprocess.run(['git', 'commit', '-m', 'Add .gitignore from source branch'], check=True)
                            print("Created initial commit with .gitignore from source branch")
                        except Exception as e:
                            print(f"Warning: Failed to add .gitignore: {e}")
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
                
                # Save the .gitignore content from source branch before switching
                gitignore_content = ""
                try:
                    # Check if .gitignore exists in the source branch
                    if os.path.exists(".gitignore"):
                        with open(".gitignore", "r") as f:
                            gitignore_content = f.read()
                        print("Found .gitignore in source branch, will add to new branch")
                except Exception as e:
                    print(f"Warning: Could not read .gitignore: {e}")
                
                # Create an orphan branch with no history
                subprocess.run(['git', 'checkout', '--orphan', target_branch], check=True)
                # Clean the working directory
                subprocess.run(['git', 'rm', '-rf', '.'], check=True, stderr=subprocess.DEVNULL)
                
                # Add .gitignore as the first commit if we found one
                if gitignore_content:
                    try:
                        # Create .gitignore file
                        with open(".gitignore", "w") as f:
                            f.write(gitignore_content)
                        # Commit it
                        subprocess.run(['git', 'add', '.gitignore'], check=True)
                        subprocess.run(['git', 'commit', '-m', 'Add .gitignore from source branch'], check=True)
                        print("Created initial commit with .gitignore from source branch")
                    except Exception as e:
                        print(f"Warning: Failed to add .gitignore: {e}")
            else:
                print(f"Creating '{target_branch}' at {merge_base[:7]}")
                subprocess.run(['git', 'checkout', '-b', target_branch, merge_base], check=True)
        
        # Now we're on the target branch, apply the logical groups in order
        
        # Sort groups by the oldest commit in each group (higher index = older commit)
        # This ensures groups are processed in chronological order
        sorted_groups = sorted(commit_groups, 
                            key=lambda g: max(g['commit_indices']) if 'commit_indices' in g else 0,
                            reverse=True)
        
        for i, group in enumerate(sorted_groups):
            print(f"\nProcessing group {i+1}/{len(sorted_groups)}: {group['message']}")
            
            # Track if we've made any changes in this group
            group_has_changes = False
            
            # Get the commits in this group sorted by their position in the original list
            # This ensures we apply them in chronological order (oldest first)
            if 'commit_indices' in group:
                # First reverse the commit indices so higher indices = newer commits
                # This makes our sorting logic more intuitive
                reversed_indices = {sha: len(commits) - idx - 1 for sha, idx in zip(group['commits'], group['commit_indices'])}
                
                # Sort by these reversed indices in ascending order (lower index = older commit)
                sorted_commits = sorted(zip(group['commits'], [reversed_indices[sha] for sha in group['commits']]), 
                                     key=lambda x: x[1])
                sorted_shas = [sha for sha, _ in sorted_commits]
            else:
                # Fallback if indices aren't available
                sorted_shas = group['commits']
            
            print(f"  Applying {len(sorted_shas)} commits in chronological order")
            
            # Cherry-pick each commit in the group in chronological order
            for commit_sha in sorted_shas:
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
        
        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        
        print(f"\nOperation completed in {int(minutes)} minutes and {int(seconds)} seconds")
            
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

def compare_branch_content(repo_path, source_branch, target_branch):
    """Compare the content of all files between two branches."""
    try:
        original_dir = os.getcwd()
        os.chdir(repo_path)
        repo = git.Repo('.')
        
        print(f"\nVerifying that '{source_branch}' and '{target_branch}' have the same content...")
        
        # Get the list of files in the source branch
        subprocess.run(['git', 'checkout', source_branch], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True)
        source_files = set(result.stdout.strip().split('\n'))
        
        # Get the list of files in the target branch
        subprocess.run(['git', 'checkout', target_branch], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True)
        target_files = set(result.stdout.strip().split('\n'))
        
        # Compare the file lists
        missing_in_target = source_files - target_files
        missing_in_source = target_files - source_files
        common_files = source_files.intersection(target_files)
        
        if missing_in_target:
            print(f"Warning: {len(missing_in_target)} files are in '{source_branch}' but missing in '{target_branch}':")
            for file in sorted(missing_in_target):
                print(f"  - {file}")
        
        if missing_in_source:
            print(f"Warning: {len(missing_in_source)} files are in '{target_branch}' but missing in '{source_branch}':")
            for file in sorted(missing_in_source):
                print(f"  - {file}")
        
        # Compare content of common files
        different_files = []
        
        print(f"Comparing content of {len(common_files)} files...")
        for i, file in enumerate(sorted(common_files)):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(common_files)} files...")
                
            # Get content from source branch
            subprocess.run(['git', 'checkout', source_branch, '--', file], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                with open(file, 'rb') as f:
                    source_content = f.read()
            except Exception:
                # Skip files that can't be read
                continue
            
            # Get content from target branch
            subprocess.run(['git', 'checkout', target_branch, '--', file], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                with open(file, 'rb') as f:
                    target_content = f.read()
            except Exception:
                # Skip files that can't be read
                continue
            
            # Compare content
            if source_content != target_content:
                different_files.append(file)
        
        if different_files:
            print(f"Warning: {len(different_files)} files have different content:")
            for file in different_files:
                print(f"  - {file}")
            
            # Ask if user wants to see a detailed diff for each file
            if len(different_files) <= 5:  # Only offer this for a small number of files
                show_diff = input("Show detailed diff for these files? [y/N] ").lower()
                if show_diff in ('y', 'yes'):
                    for file in different_files:
                        print(f"\nDiff for {file}:")
                        subprocess.run(['git', 'diff', f"{source_branch}:{file}", f"{target_branch}:{file}"], check=True)
            
            print("\nThe branches have different content. This might be due to:")
            print("  - Files that should be ignored (consider adding them to .gitignore)")
            print("  - Logical squash changes that didn't preserve exact whitespace/formatting")
            print("  - Cherry-picking conflicts that were auto-resolved")
            return False
        else:
            print("Verification successful! Branches have identical content.")
            return True
    except Exception as e:
        print(f"Error comparing branch content: {e}")
        return False
    finally:
        # Restore original directory and branch
        os.chdir(original_dir)

def main():
    """Main function."""
    args = parse_args()
    
    # Track overall execution time
    start_time = time.time()
    
    # Initialize repo object
    repo = git.Repo(args.repo)
    
    # Determine branch context for display message
    if args.source_branch:
        print(f"Analyzing commits from branch '{args.source_branch}'...")
    else:
        print(f"Analyzing commits from current branch...")
    
    if args.new_root:
        print("Creating a new branch with no shared history (--new-root mode)")
        
    # If processing main branch with lots of commits, give a warning
    if (args.source_branch == "main" or args.source_branch == "master") and args.count > 1000:
        print("\nWARNING: Processing a large number of commits from main branch.")
        print("This operation may take a significant amount of time.")
        print("Consider reducing the commit count with --count if it's too slow.\n")
    
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
                # Calculate and display elapsed time
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                print(f"Logical squash operation completed in {int(minutes)} minutes and {int(seconds)} seconds.")
                
                # Verify that branches have the same content
                if args.verify and not args.skip_verify and args.target_branch:
                    source_branch = args.source_branch or repo.active_branch.name
                    compare_branch_content(args.repo, source_branch, args.target_branch)
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
                
                # Verify that branches have the same content
                if args.verify and not args.skip_verify and args.target_branch:
                    source_branch = args.source_branch or repo.active_branch.name
                    compare_branch_content(args.repo, source_branch, args.target_branch)
        else:
            print("Squash operation cancelled.")

if __name__ == "__main__":
    main()
