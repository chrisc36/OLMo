import math
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from os import listdir
from os.path import join

import seaborn
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def check_layer1():
    src = "/Users/chrisc/data/dbg-olmo-7b-grad-norm"
    names = ["att_proj", "attn_out", "ff_out", "ff_proj"]
    grads = {}
    for name in names:
        with open(f"{src}/_fsdp_wrapped_module.transformer.blocks.1._fsdp_wrapped_module.{name}.weight.pkl", "rb") as f:
            grads[name] = pickle.load(f)
    for name, grad in grads.items():
        # if name != "att_proj":
        #     continue
        grad = grad*grad
        map = seaborn.heatmap(np.power(-np.sqrt(grad)))
        plt.title(name)
        plt.savefig(f"/Users/chrisc/Desktop/{name}.png")
        plt.show()
        for dim in [0, 1]:
            if name == "ff_proj" or name == "att_proj":
                dim = 1 - dim
            norms = np.sqrt(grad.sum(dim))
            for k in np.argsort(-norms)[:10]:
                print(name, dim, k, norms[k])
            plt.plot(norms)
            plt.title(f"{name}-{dim}-{tuple(grad.shape)}")
            plt.savefig(f"/Users/chrisc/Desktop/plots/{name}-{dim}-{tuple(grad.shape)}")
            plt.show()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _load(file):
    step, file = file
    with open(file, "rb") as f:
        _, data, _ = pickle.load(f)
        data["loss"] = float(data["loss"])
        return step, data


def per_tokens():
    src = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/from-step536000-r9-examples-loss-mask"
    src = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/from-step536000-r9-batch0-seq"

    with open(join(src, "step0.pkl"), "rb") as f:
        batch = pickle.load(f)[-1]

    other_stats = defaultdict(float)
    ex416_stats = None
    other_count = 9
    grads = []
    losses = []
    steps = []
    targets = [x for x in listdir(src) if x.endswith(".pkl")]
    targets = [(int(x[4:].split(".")[0]), join(src, x)) for x in targets]
    targets.sort(key=lambda x: x[0])

    # with Pool(processes=12) as pool:
    #     for step, data in tqdm(pool.imap_unordered(_load, targets), total=len(targets), ncols=100, smoothing=0.5):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for step, data in tqdm(pool.map(_load, targets), total=len(targets), ncols=100, smoothing=0.5):
            steps.append(step)
            grads.append(data["total_grad_norm"])

    steps = np.array(steps)
    grads = np.array(grads)
    ix = np.argsort(steps)
    steps = steps[ix]
    grads = grads[ix]
    for i in np.argsort(-grads)[:10]:
        print(i, grads[i])


    plt.plot(steps, grads)
    plt.savefig("/Users/chrisc/Desktop/plots/grads.png")
    plt.show()

    plt.plot(steps[2500:2600], grads[2500:2600])
    plt.savefig("/Users/chrisc/Desktop/plots/grads_subset.png")
    plt.show()


def per_example():
    src = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/r3-step543041-per-example"
    other_stats = defaultdict(float)
    high_grad_stats = None
    other_count = 0
    grads = []
    losses = []
    targets = [x for x in listdir(src) if x.endswith(".pkl")]
    targets = [(int(x[4:].split(".")[0]), join(src, x)) for x in targets]
    targets = sorted(targets, key=lambda x: x[0])[300:400]

    # with Pool(processes=4) as pool:
    #     for step, data in tqdm(pool.imap_unordered(_load, targets), total=len(targets), ncols=100, smoothing=0.5):
    # with ThreadPoolExecutor(max_workers=4) as pool:
    #     for step, data in tqdm(pool.map(_load, targets), total=len(targets), ncols=100, smoothing=0.5):
    for step, target in tqdm(targets):
            step, data = _load((step, target))
            grads.append(data["total_grad_norm"])
            data["loss"] = float(data["loss"])
            if step == 340:
                high_grad_stats = data
            else:
                for k, v in data.items():
                    other_stats[k] += v
                other_count += 1
            grads.append(data["total_grad_norm"])
            losses.append(data["loss"])

    assert high_grad_stats is not None
    other_stats = {k: v/other_count for k, v in other_stats.items()}

    # for param in ["att_proj", "attn_out", "ff_out", "ff_proj"]:
    #     x = list(range(32))
    #     key = [f"grad/transformer.blocks.{i}.{param}.weight.norm" for i in x]
    #     plt.plot([high_grad_stats[x] for x in key], label=param)
    #     plt.plot([other_stats[x] for x in key], label=param)
    #     if len(x) < 10:
    #         plt.xticks(x)
    #     plt.title(param)
    #     plt.savefig(f"/Users/chrisc/Desktop/plots/{param}.png")
    #     plt.show()

    plt.figure(figsize=(12, 4))
    for param in ["att_proj", "attn_out", "ff_out", "ff_proj"]:
        x = list(range(5))
        key = [f"grad/transformer.blocks.{i}.{param}.weight.norm" for i in x]
        plt.plot([high_grad_stats[x] for x in key], label=param)
        plt.plot([other_stats[x] for x in key], label=param)
        if len(x) < 10:
            plt.xticks(x)
        plt.title(param)
        plt.savefig(f"/Users/chrisc/Desktop/plots/{param}.png")
        plt.show()

    # plt.plot(grads)
    # plt.savefig("/Users/chrisc/Desktop/plots/grads.png")
    # plt.show()
    # plt.plot(losses)
    # plt.savefig("/Users/chrisc/Desktop/plots/losses.png")
    # plt.show()


def compare_per_example_grad_norm():
    src1 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r4-stats"
    src2 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r3-stats"
    all_grads = []
    all_steps = []
    for src in [src1, src2]:
        targets = [x for x in listdir(src) if x.endswith(".pkl")]
        targets = [(int(x[4:].split(".")[0]), join(src, x)) for x in targets]
        targets = sorted(targets, key=lambda x: x[0])
        # targets = targets[128:256]
        grads = []
        steps = []
        for step, target in tqdm(targets):
            step, data = _load((step, target))
            grads.append(data["total_grad_norm"])
            steps.append(step)
        grads = np.array(grads)
        steps = np.array(steps)
        ix = np.argsort(steps)
        grads = grads[ix]
        all_grads.append(grads)
        all_steps.append(np.array(steps))

    # for ix, src in enumerate(all_grads):
    #     plt.plot(src)
    #     plt.savefig(f"/Users/chrisc/Desktop/plots/grads{ix}.png")
    #     plt.show()
    plt.figure(figsize=(16, 4))
    for ix, src in enumerate(all_grads):
        marker = "+" if ix == 0 else "x"
        plt.scatter(all_steps[0], np.minimum(src, 20), alpha=0.6, lw=1, marker=marker)
    plt.savefig(f"/Users/chrisc/Desktop/plots/grads_joint.png")
    plt.show()


def compare_stats():
    src1 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r4-stats"
    src2 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r3-stats"
    data = []
    for src in [src1, src2]:
        with open(join(src, "step211.pkl"), "rb") as f:
            dbg_stats, opt_stats, _ = pickle.load(f)
            dbg_stats.update(opt_stats)
            data.append(dbg_stats)
    high_grad_stats, other_stats = data

    # for param in ["att_proj", "attn_out", "ff_out", "ff_proj"]:
    #     x = list(range(32))
    #     key = [f"grad/transformer.blocks.{i}.{param}.weight.norm" for i in x]
    #     plt.plot([high_grad_stats[x] for x in key], label=param)
    #     plt.plot([other_stats[x] for x in key], label=param)
    #     if len(x) < 10:
    #         plt.xticks(x)
    #     plt.title(param)
    #     plt.savefig(f"/Users/chrisc/Desktop/plots/{param}.png")
    #     plt.show()
    #
    for param in ["input-norm", "att-norm", "post-att-norm", "ff-norm", "post-ff-norm"]:
        x = list(range(32))
        key = [f"layer{i}-{param}" for i in x]
        plt.plot([high_grad_stats[x] for x in key], label=param)
        plt.plot([other_stats[x] for x in key], label=param)
        if len(x) <= 10:
            plt.xticks(x)
        plt.title(param)
        plt.savefig(f"/Users/chrisc/Desktop/plots/{param}.png")
        plt.show()

    # for param in ["input-max", "post-att-max", "att-max", "ff-max", "post-ff-max"]:
    #     x = list(range(32))
    #     key = [f"layer{i}-{param}" for i in x]
    #     plt.plot([torch.tensor(high_grad_stats[x], dtype=torch.float32) for x in key], label=param)
    #     plt.plot([torch.tensor(other_stats[x], dtype=torch.float32) for x in key], label=param)
    #     if len(x) < 10:
    #         plt.xticks(x)
    #     plt.title(param)
    #     plt.savefig(f"/Users/chrisc/Desktop/plots/{param}.png")
    #     plt.show()


def compare_clipping_rates():
    src1 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r4-step211-with-clip.pkl"
    src2 = "/Users/chrisc/data/dbg-olmo-7b-grad-norm/mitchish7-datafix/r7-step211-with-clip.pkl"

    data1 = load_pickle(src1)[1]
    data2 = load_pickle(src2)[1]

    for layer in range(32):
        print()


if __name__ == '__main__':
    compare_clipping_rates()