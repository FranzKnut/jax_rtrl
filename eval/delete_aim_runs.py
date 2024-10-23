"""Remove unwanted aim runs."""
import argparse
import aim

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=str, default=".")
parser.add_argument('--remove', action='store_true')
args = parser.parse_args()

repo = aim.Repo(args.repo)
query = "run.created_at.day < 10"

runs = repo.query_runs(query)
hashes = [r.run.hash for r in runs.iter_runs()]

print(f"Found {len(hashes)} runs")

if args.remove:
    print("REMOVING...")
    repo.delete_runs(hashes)
    print("done.")
