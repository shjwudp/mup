import argparse
import copy
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mup.coord_check import get_coord_data, plot_coord_data
from mup import set_base_shapes

from .model import GPT, GPTConfig
from .dataset import CharDataset
from .trainer import TrainerConfig


###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchloader(train_data, bptt):
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        yield get_batch(train_data, i, bptt)


def coord_check(args, mup):
    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size=128)

    def gen(w, standparam=False):
        def f():
            model = GPT(GPTConfig(
                train_dataset.vocab_size,
                train_dataset.block_size,
                n_layer=8,
                n_head=8,
                n_embd=512,
            ))
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, args.load_base_shapes)
            return model
        return f

    optimizer = copy.deepcopy(args.optimizer)
    optimizer = optimizer.replace("mu", "")

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(models, batchloader(train_dataset, args.bptt), mup=mup, lr=lr, optimizer=optimizer, flatten_output=True, nseeds=nseeds, nsteps=nsteps, lossfn='nll')


def train(args):
    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size=128)

    model = GPT(GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=8,
        n_head=8,
        n_embd=512,
    ))
    tconf = TrainerConfig(max_epochs=2, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*train_dataset.block_size,
                      num_workers=4)
    optimizer = model.configure_optimizers(tconf)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=args.batch_size,
    )
    losses = []
    for it, (x, y) in enumerate(train_loader):
        logits, loss = model(x, y)
        loss = loss.mean()
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        if it % args.log_interval == 0:
            logging.info("loss: {}".format(float(np.mean(losses))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'musgd', 'adam', 'muadam'])
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument("--log_interval", default=10, type=int)

    args = parser.parse_args()

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    train(args)


if __name__ == "__main__":
    main()
