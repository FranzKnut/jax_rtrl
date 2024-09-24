"""Loop over all runs and do something."""

import argparse
from tqdm import tqdm
import aim


parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--repo", default="", type=str)
parser.add_argument("--query", default="", type=str)
parser.add_argument("--version", default=1)
args = parser.parse_args()

repo = aim.Repo(args.repo)
# Get the hashes of all runs that match the query, but don't have a version set
hashes = []
for r in repo.query_runs(args.query).iter_runs():
    print(r.run.hash, end=" ")
    if r.run.get("fix_version", None) is None or args.force:
        hashes.append(r.run.hash)
        print("*")
    else:
        print()
failed_count = 0
print("Fix version:", args.version)
for h in tqdm(hashes):
    tqdm.write(h)
    try:
        # Get a writeable run object
        run = aim.Run(h, repo=args.repo)
        # Set version so we know it was edited
        run["fix_version"] = args.version
        # DO SOMETHING TO RUNS HERE

        edit = run["hparams"]

        if edit.get("obs_mask", None) is None:
            edit["obs_mask"] = "None"

        if edit.get("agent_type", None) in [None, "rnn"]:
            edit["agent_type"] = "rflo"

        # WRITE CHANGES
        run["hparams"] = edit
        run.close()
    except:  # noqa
        failed_count += 1
        tqdm.write(f"Failed! {failed_count} failed so far.")

if failed_count > 0:
    print(f"Failed to fix {failed_count} runs.")

print("done.")
