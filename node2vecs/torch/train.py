import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader


def train(
    model,
    dataset,
    loss_func,
    batch_size=10000,
    device="cpu",
    checkpoint=10000,
    outputfile=None,
    learning_rate=1e-3,
):
    # Set the device parameter if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    #
    # Set up the model
    #
    model.train()
    model = model.to(device)

    # Training
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    # optim = Adam(focal_params, lr=0.003)
    optim = AdamW(focal_params, lr=learning_rate)

    pbar = tqdm(dataloader, miniters=100, total=len(dataloader))
    it = 0
    for params in pbar:
        # clear out the gradient
        focal_params = filter(lambda p: p.requires_grad, model.parameters())
        for param in focal_params:
            param.grad = None

        for i, p in enumerate(params):
            params[i] = p.to(device)

        # compute the loss
        loss = loss_func(*params)

        # backpropagate
        loss.backward()

        # update the parameters
        optim.step()

        pbar.set_postfix(loss=loss.item())

        if (it + 1) % checkpoint == 0:
            if outputfile is not None:
                torch.save(model.state_dict(), outputfile)
        it += 1

    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model
