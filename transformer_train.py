"""
This file enables https://pytorch.org/memory_viz
"""

import torch, socket, datetime
from torch import nn
from torch.utils.data import DataLoader
from transformer_model import Transformer
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.autograd.profiler import record_function

TIME_FORMAT_STR: str = "%b_%d_%H_%m_%s"


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


def trace_handler(prof: torch.profiler.profile):
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    prof.export_memory_timeline(f"{file_prefix.html}", device="cuda:0")


model = Transformer().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


training_data = datasets.FashionMNIST(
    root="data", train=True, download=False, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=False, transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")


def train(dataloader, model, loss_fn, optimizer, mem_profile=False):
    size = len(dataloader.dataset)

    if mem_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        ) as prof:
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                prof.step()

                with record_function("## Forward ##"):
                    # Compute prediction error
                    pred = model(X)
                    loss = loss_fn(pred, y)

                with record_function("## Backward ##"):
                    # Backpropagation
                    loss.backward()

                with record_function("## Optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    else:
        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(
        f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    if 10 > epochs > 3:
        print(f"Memory profiling enabled")
        train(train_dataloader, model, loss_fn, optimizer, mem_profile=True)
    else:
        train(train_dataloader, model, loss_fn, optimizer, mem_profile=False)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
