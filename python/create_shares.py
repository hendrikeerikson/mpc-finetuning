import sharemind as sm

import numpy as np
import torch

from torchvision import datasets
from torchvision import transforms

from backbone import BackboneMLP

# SMTYPE="fix64"
# SMTYPE="float32"
SMTYPE="ff16"
# NPTYPE=np.float64
NPTYPE=np.float32


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


def store(party_id, name, share):
    with open(f"./data/p{party_id}/{name}.bin", 'wb') as f:
        f.write(share)


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

    classifier_W = share_tensor(ctrl, model.classifier.weight)
    classifier_b = share_tensor(ctrl, model.classifier.bias)

    for i in [0,1,2]:
        store(i+1, "fc1_W", fc1_W[i])
        store(i+1, "fc1_b", fc1_b[i])
        store(i+1, "fc2_W", fc2_W[i])
        store(i+1, "fc2_b", fc2_b[i])
        store(i+1, "classifier_W", classifier_W[i])
        store(i+1, "classifier_b", classifier_b[i])
        store(i+1, "x_train", x_train_shares[i])
        store(i+1, "y_train", y_train_shares[i])
        store(i+1, "x_test", x_test_shares[i])
        store(i+1, "y_test", y_test_shares[i])


if __name__ == "__main__":
    load_arguments()

