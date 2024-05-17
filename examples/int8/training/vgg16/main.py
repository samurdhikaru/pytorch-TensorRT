import argparse
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vgg16 import vgg16

PARSER = argparse.ArgumentParser(
    description="VGG16 example to use with Torch-TensorRT PTQ"
)
PARSER.add_argument(
    "--epochs", default=100, type=int, help="Number of total epochs to train"
)
PARSER.add_argument(
    "--batch-size", default=128, type=int, help="Batch size to use when training"
)
PARSER.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
PARSER.add_argument("--drop-ratio", default=0.0, type=float, help="Dropout ratio")
PARSER.add_argument("--momentum", default=0.9, type=float, help="Momentum")
PARSER.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
PARSER.add_argument(
    "--fp8-epochs",
    default=0,
    type=int,
    help="Enable FP8 and specify the number of epochs after the regular training to quantize the model to FP8",
)
PARSER.add_argument(
    "--ckpt-dir",
    default="/tmp/vgg16_ckpts",
    type=str,
    help="Path to save checkpoints (saved every 10 epochs)",
)
PARSER.add_argument(
    "--start-from",
    default=0,
    type=int,
    help="Epoch to resume from (requires a checkpoin in the providied checkpoi",
)
PARSER.add_argument("--seed", type=int, help="Seed value for rng")
PARSER.add_argument(
    "--tensorboard",
    type=str,
    default="/tmp/vgg16_logs",
    help="Location for tensorboard info",
)

args = PARSER.parse_args()
for arg in vars(args):
    print(" {} {}".format(arg, getattr(args, arg)))
state = {k: v for k, v in args._get_kwargs()}

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print("RNG seed used: ", args.seed)

now = datetime.now()

timestamp = datetime.timestamp(now)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    global state
    global classes
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    training_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    testing_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    testing_dataloader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    num_classes = len(classes)

    model = vgg16(num_classes=num_classes, init_weights=False)
    model = model.cuda()

    data = iter(training_dataloader)
    images, _ = next(data)

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.start_from != 0:
        ckpt_file = args.ckpt_dir + "/ckpt_epoch" + str(args.start_from) + ".pth"
        print("Loading from checkpoint {}".format(ckpt_file))
        assert os.path.isfile(ckpt_file)
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        state = ckpt["state"]

    for epoch in range(args.start_from, args.epochs):
        adjust_lr(opt, epoch)
        print("Epoch: [%5d / %5d] LR: %f" % (epoch + 1, args.epochs, state["lr"]))

        train(model, training_dataloader, crit, opt, epoch)
        test_loss, test_acc = test(model, testing_dataloader, crit, epoch)

        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))

        if epoch % 10 == 9 or epoch == args.epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "acc": test_acc,
                    "opt_state_dict": opt.state_dict(),
                    "state": state,
                },
                ckpt_dir=args.ckpt_dir,
            )

    if args.fp8_epochs > 0:
        print("[PTQ] Quantizing model to FP8...")
        import modelopt.torch.quantization as mtq
        import torch_tensorrt as torchtrt
        from modelopt.torch.quantization.utils import export_torch_mode

        def calibrate_loop(model):
            # calibrate on a small number of batches
            for fp8_ep in range(args.fp8_epochs):
                print("Epoch: [%5d / %5d]" % (fp8_ep + 1, args.fp8_epochs))
                total = 0
                correct = 0
                loss = 0.0
                for data, labels in training_dataloader:
                    data, labels = data.cuda(), labels.cuda(non_blocking=True)
                    out = model(data)
                    loss += crit(out, labels)
                    preds = torch.max(out, 1)[1]
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                print(
                    "Test Loss: {:.5f} Test Acc: {:.2f}%".format(
                        loss / total, 100 * correct / total
                    )
                )

        quant_cfg = mtq.FP8_DEFAULT_CFG
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
        # model has FP8 qdq nodes at this point
        with torch.no_grad():
            with export_torch_mode():
                input_tensor = images.cuda()
                exp_program = torch.export.export(model, (input_tensor,))
                trt_model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[input_tensor],
                    enabled_precisions={torch.float8_e4m3fn},
                    min_block_size=1,
                    debug=False,
                )
                outputs_trt = trt_model(input_tensor)
                print("TRT outputs:\n", outputs_trt)


def train(model, dataloader, crit, opt, epoch):
    model.train()
    running_loss = 0.0
    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if batch % 50 == 49:
            print(
                "Batch: [%5d | %5d] loss: %.3f"
                % (batch + 1, len(dataloader), running_loss / 100)
            )
            running_loss = 0.0


def test(model, dataloader, crit, epoch):
    global classes
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds, epoch)
    # print(loss, total, correct, total)
    return loss / total, correct / total


def save_checkpoint(state, ckpt_dir="checkpoint"):
    print("Checkpoint {} saved".format(state["epoch"]))
    filename = "ckpt_epoch" + str(state["epoch"]) + ".pth"
    filepath = os.path.join(ckpt_dir, filename)
    torch.save(state, filepath)


def adjust_lr(optimizer, epoch):
    global state
    new_lr = state["lr"] * (0.5 ** (epoch // 40)) if state["lr"] > 1e-7 else state["lr"]
    if new_lr != state["lr"]:
        state["lr"] = new_lr
        print("Updating learning rate: {}".format(state["lr"]))
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    global classes
    """
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]


if __name__ == "__main__":
    main()
