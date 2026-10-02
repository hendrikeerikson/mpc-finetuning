import sharemind as sm

import asyncio
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from torchvision import datasets
from torchvision import transforms

from backbone import BackboneMLP


# BYTECODE="bytecode/bench_lastlayer_fix64.sb"
BYTECODE="bytecode/bench_lastlayer_ff16.sb"
# BYTECODE="bytecode/bench_lastlayer_float32.sb"
# SMTYPE="fix64"
# SMTYPE="float32"
SMTYPE="ff16"
# NPTYPE=np.float64
NPTYPE=np.float32


def run(party_id, inputs):
    args = {
        1: (1, "127.0.0.1:4003", "0.0.0.0:4001"),
        2: (2, "127.0.0.1:4001", "0.0.0.0:4002"),
        3: (3, "127.0.0.1:4002", "0.0.0.0:4003"),
    }

    rt = sm.new_shared3p(*args[party_id])
    sm.execute( rt, BYTECODE, inputs )

    print("finished execute", party_id)
    return


def build_dataset(classes, train):
    ds = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    xs = []
    ys = []

    class_offset = min(classes)
    num_classes = len(classes)

    for x, y in ds:

        if y not in classes:
            continue

        x = x.view(-1).numpy()

        # Convert labels: [8,9] -> [0,1]
        y = int(y) - class_offset

        xs.append(x)
        ys.append(y)

    xs = np.asarray( xs, dtype=NPTYPE,)
    ys = np.asarray( ys, dtype=np.int64,)

    ys_one_hot = np.eye(num_classes, dtype=NPTYPE)[ys]
    return xs, ys_one_hot


def share_tensor(pd, tensor):
    arr = tensor.detach().cpu().numpy()
    arr_fp32 = arr.astype(NPTYPE)
    shares = sm.share(pd, SMTYPE, arr_fp32.tobytes())

    print(arr_fp32.shape, len(shares), len(shares[0]))

    return shares


def share_ndarray(pd, arr):
    shares = sm.share(pd, SMTYPE, arr.tobytes())
    print(arr.shape, len(shares), len(shares[0]))

    return shares


def load_arguments():
    classes = [8, 9]
    model = BackboneMLP(
        hidden1=cfg["model"]["hidden1"],
        hidden2=cfg["model"]["hidden2"],
        num_classes=len(classes)
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/pretrained.pt",
            map_location="cpu",
        )
    )

    ctrl = sm.new_shared3p_ctrl()
    arg_map = [{}, {}, {}]

    fc1_W = share_tensor(ctrl, model.fc1.weight)
    fc1_b = share_tensor(ctrl, model.fc1.bias)
    fc2_W = share_tensor(ctrl, model.fc2.weight)
    fc2_b = share_tensor(ctrl, model.fc2.bias)

    x_train, y_train = build_dataset( classes, train=True )
    x_train_shares = share_ndarray(ctrl, x_train)
    y_train_shares = share_ndarray(ctrl, y_train)

    x_test, y_test = build_dataset( classes, train=False )
    x_test_shares = share_ndarray(ctrl, x_test[0:100,:])
    y_test_shares = share_ndarray(ctrl, y_test[0:100,:])

    for i in [0,1,2]:
        arg_map[i]["fc1_W"] = fc1_W[i]
        arg_map[i]["fc1_b"] = fc1_b[i]
        arg_map[i]["fc2_W"] = fc2_W[i]
        arg_map[i]["fc2_b"] = fc2_b[i]
        arg_map[i]["x_train"] = x_train_shares[i]
        arg_map[i]["y_train"] = y_train_shares[i]
        arg_map[i]["x_test"] = x_test_shares[i]
        arg_map[i]["y_test"] = y_test_shares[i]

    return arg_map


async def main():
    loop = asyncio.get_running_loop()

    arg_map = load_arguments()

    # Run functions in isolated processes rather than threads
    with ProcessPoolExecutor() as pool:
        await asyncio.gather(
            loop.run_in_executor(pool, run, 1, arg_map[0] ),
            loop.run_in_executor(pool, run, 2, arg_map[1] ),
            loop.run_in_executor(pool, run, 3, arg_map[2] ))

if __name__ == "__main__":
    asyncio.run(main())

