import torch
from network import ProgressiveSiren
from torch.utils.data import DataLoader
from tqdm import trange
from utils.evaluation import *


class SDFTrainer():
    def __init__(self):
        None

    def train_progressive(self, args, net, dataset, result_dir, loss_fn, evaluate):
        '''
        <predefined widths used in the paper>
        SDF spectral growing: 148, 240, 309
        '''
        print("streamable (progressive training)")

        dataloader = DataLoader(dataset, shuffle = True, batch_size = 1, pin_memory = True, num_workers = 0)
        n_points = dataset.n_points

        for w_idx, width in enumerate(args.widths):
            net.train()
            net.runnable_widths.append(width)
            if w_idx != 0:
                net.grow_width(width = args.widths[w_idx] - args.widths[w_idx - 1])

            print(f'current width: {width}')
            print(f'number of parameters: {sum(p.numel() for p in net.parameters()) - net.discarded_weight}')

            net.select_subnet(w_idx)
            optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

            best_loss = 10000.
            for e in trange(1, args.epochs+1):
                total_loss = 0.
                for step, (x, gt) in enumerate(dataloader):
                    x = {key: value.cuda() for key, value in x.items()}
                    x = x['coords'].requires_grad_()
                    yhat = net(x)
                    losses = loss_fn(yhat, gt, x)
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                    optimizer.zero_grad()
                    train_loss.backward()
                    total_loss += train_loss
                    if w_idx > 0:
                        net.freeze_subnet(w_idx - 1)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.)
                    optimizer.step()

                # save best model
                if total_loss < best_loss:
                    best_loss = total_loss.item()
                    torch.save(net.state_dict(), os.path.join(result_dir, 'model.pth'))

                if e % args.log_iter == 0:
                    print(f"EPOCH: {e}/{args.epochs} WIDTH: {width} LOSS: {train_loss.item()}")

            with torch.no_grad():
                net.eval()
                evaluate(net, w_idx)

        print("training done")


    def train_individual(self, args, device, dataset, result_dir, loss_fn, evaluate):
        '''
        <predefined widths used in the paper>
        SDF spectral growing: 148, 210, 256
        '''
        print("training individual model")

        dataloader = DataLoader(dataset, shuffle = True, batch_size = 1, pin_memory = True, num_workers = 0)

        for w_idx, width in enumerate(args.widths):
            print(f'current width: {width}')
            net=ProgressiveSiren(
                in_feats=dataset.x.shape[-1],
                hidden_feats=width,
                n_hidden_layers=args.n_hidden_layers,
                out_feats=dataset.y.shape[-1],
                device=device).to(device)
            print(f'number of parameters: {sum(p.numel() for p in net.parameters())}')

            optimizer=torch.optim.Adam(net.parameters(), lr=args.lr)
            net.train()
            net.select_subnet(0)
            best_loss = 10000.
            for e in trange(args.epochs):
                total_loss = 0.
                for step, (x, gt) in enumerate(dataloader):
                    x = {key: value.cuda() for key, value in x.items()}
                    x = x['coords'].requires_grad_()
                    yhat = net(x)
                    losses = loss_fn(yhat, gt, x)
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                    optimizer.zero_grad()
                    train_loss.backward()
                    total_loss += train_loss
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.)
                    optimizer.step()

                # save best model
                if total_loss < best_loss:
                    best_loss = total_loss.item()
                    torch.save(net.state_dict(), os.path.join(result_dir, 'model.pth'))

                if e % args.log_iter == 0:
                    print(f"EPOCH: {e}/{args.epochs} WIDTH: {width} LOSS: {train_loss.item()}")

            with torch.no_grad():
                net.eval()
                evaluate(net, w_idx)

        print("training done")