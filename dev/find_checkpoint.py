import wandb
import numpy as np
import re


def main():
    # checkpoints = {}
    # with open("steps.txt") as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         assert line.startswith("PRE ")
    #         line = line[4:]
    #         line = line.rstrip("/")
    #         if "huggingface" in line:
    #             continue
    #         if "unsharded" in line:
    #             continue
    #         checkpoints[int(line[4:])] = line
    # all_steps = sorted(checkpoints)

    api = wandb.Api()
    # runs = api.runs(
    #     "ai2-llm/olmo-medium",
    #     filters={"group":  "mitchish7"}
    # )
    runs = [api.run("ai2-llm/olmo-medium/b33slso9")]

    entity_id_to_step = {}
    for run in runs:
        load_path = run.config["load_path"]
        if load_path is None:
            first_step = 0
        else:
            first_step = int(re.match(".*step([0-9]+)", run.config["load_path"]).group(1))
        entity_id_to_step[run.id] = first_step
    runs = sorted(runs, key=lambda x: entity_id_to_step[x.id])

    # has_checkpoint = np.array(list(checkpoints.keys()), dtype=np.int64)
    # np.sort(has_checkpoint)
    max_gnorm = -1
    max_checkpoint = None
    for run in runs:
        first_step = entity_id_to_step[run.id]
        if first_step < 200000:
            continue

        history = run.scan_history(page_size=500, keys=["_step", "optim/total_grad_norm"], min_step=first_step)
        for ix, row in enumerate(history):
            step_gnorm = row["optim/total_grad_norm"]
            step = row["_step"]
            # ix = np.searchsorted(has_checkpoint, step, side="right") - 1
            # checkpoint_step = has_checkpoint[ix]
            checkpoint_step = (step // 1000)* 1000
            diff = step - checkpoint_step  # How many steps to get from a checkpoint to `step`
            assert diff >= 0
            if diff == 0:
                ix =- 1
                diff = step - checkpoint_step
            if diff <= 30:
                step_gnorm = row["optim/total_grad_norm"]
                if step_gnorm > max_gnorm:
                    max_gnorm = step_gnorm
                    max_checkpoint = step
                if step_gnorm == max_gnorm or step_gnorm > 1.0:
                    print(step_gnorm, step, checkpoint_step)
            if step % 10000 == 0:
                print(step)


def copy_wandb():
    api = wandb.Api()
    entity = "ai2-llm"
    project = "olmo-medium"

    # Get the runs from the source project
    run = api.run(f"{entity}/{project}/0o2xzqba")
    print(run)
    # runs = api.runs(
    #     "ai2-llm/olmo-medium",
    #     filters={"group":  "mitchish7"}
    # )
    # print(len(runs))
    # return
    history = run.scan_history(page_size=500, min_step=543000, max_step=543050)

    # Create a new run in the destination project
    new_run = wandb.init(
        project=project, entity=entity, config=run.config, name=run.name,
        group="dbg2-mitchish7-from543000",
        resume="allow"
    )

    # Log the history to the new run
    for ix, row in enumerate(history):
        row.pop("_runtime")
        row.pop("_timestamp")
        new_run.log(row, step=row.pop("_step"))
        if ix % 100 == 0:
            print(f"Adding row {ix}")

    new_run.finish()


if __name__ == '__main__':
    main()