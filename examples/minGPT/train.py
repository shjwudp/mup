import argparse
import copy
import os
import sys
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes
from mup import set_base_shapes

from model import GPT, GPTConfig
from dataset import CharDataset
from trainer import TrainerConfig


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


def coord_check(args, mup, plotdir='', legend=False):
    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size=args.bptt)

    def gen(w, standparam=False):
        def f():
            model = GPT(GPTConfig(
                train_dataset.vocab_size,
                train_dataset.block_size,
                n_layer=8,
                n_head=8,
                n_embd=w,
            ))
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, args.load_base_shapes)
            return model
        return f

    optimizer = copy.deepcopy(args.optimizer)
    optimizer = optimizer.replace("mu", "")

    widths = 2 ** np.arange(7, 12)
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=args.batch_size,
    )
    df = get_coord_data(models, train_loader, mup=mup, lr=args.lr, optimizer=optimizer,
        nseeds=args.coord_check_nseeds, nsteps=args.coord_check_nsteps)

    prm = 'μP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_trsfmr_{optimizer}_coord.png'),
        suptitle=f'{prm} Transformer {optimizer} lr={args.lr} nseeds={args.coord_check_nseeds}',
        face_color='xkcd:light grey' if not mup else None)


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
    tconf = TrainerConfig(max_epochs=2, batch_size=args.batch_size, learning_rate=6e-4,
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


def save_base_shapes(args):
    text = open("input.txt", "r").read()
    train_dataset = CharDataset(text, block_size=128)

    model = GPT(GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=args.nlayers,
        n_head=args.nhead,
        n_embd=args.d_model,
    ))
    print(f'saving base shapes at {args.save_base_shapes}')
    base_shapes = get_shapes(model)
    delta_shapes = get_shapes(
        GPT(GPTConfig(
            train_dataset.vocab_size,
            train_dataset.block_size,
            n_layer=args.nlayers,
            n_head=args.nhead,
            n_embd=args.d_model * 2,
        ))
    )
    make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
    print("done and exit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--d_model', type=int, default=256,
                        help='width of the model')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'musgd', 'adam', 'muadam'])
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--coord_check', action='store_true',
                        help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    parser.add_argument('--coord_check_nsteps', type=int, default=3,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=3,
                        help='number of seeds for testing correctness of μ parametrization')
    args = parser.parse_args()

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    if args.save_base_shapes:
        save_base_shapes(args)
        sys.exit()

    if args.coord_check:
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(args, mup=False, plotdir=plotdir, legend=False)
        sys.exit()

    train(args)


if __name__ == "__main__":
    main()
