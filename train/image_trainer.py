import torch
from tqdm import trange

from network import ProgressiveSiren
from utils.evaluation import *


class ImageTrainer():
    def __init__(self):
        None

    def train_progressive(self, args, net, dataset, result_dir, loss_fn, evaluate):
        '''
        <predefined widths used in the paper>
        image spectral growing: 30, 69, 110, 151
        image spatial growing: 59, 96, 126, 149
        '''
        print("streamable (progressive training)")
        x=dataset.x
        y=dataset.y

        for w_idx, width in enumerate(args.widths):
            net.train()
            net.runnable_widths.append(width)
            if w_idx != 0:
                net.grow_width(width = args.widths[w_idx] - args.widths[w_idx - 1])

            print(f'current width: {width}')
            print(f'number of parameters: {sum(p.numel() for p in net.parameters()) - net.discarded_weight}')

            net.select_subnet(w_idx)
            optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

            if args.experiment in ['image_spatial', 'video_temporal']:
                y=dataset.mask(input=dataset.y, w_idx=w_idx, w_num=len(args.widths))

            for e in trange(1, args.epochs+1):
                optimizer.zero_grad()
                yhat = net(x)
                loss = loss_fn(yhat, y)
                loss.backward()
                if w_idx > 0:
                    net.freeze_subnet(w_idx - 1)
                optimizer.step()

                if e % args.log_iter == 0:
                    print(f"EPOCH: {e}/{args.epochs} WIDTH: {width} LOSS: {loss.item()}")

            with torch.no_grad():
                net.eval()
                evaluate(net, w_idx)

        print("training done")
        with torch.no_grad():
            torch.save(net.state_dict(), os.path.join(result_dir, 'model.pth'))
            evaluate.metrics(n=len(net.runnable_widths))


    def train_slimmable(self, args, net, dataset, result_dir, loss_fn, evaluate):
        '''
        <predefined widths used in the paper>
        image spectral growing: 30, 60, 90, 120
        image spatial growing: 59, 84, 104, 120
        '''
        x = dataset.x
        y = dataset.y
        print("streamable (slimmable training)")
        print(f'number of TOTAL parameters: {sum(p.numel() for p in net.parameters())}')
        
        net.runnable_widths = args.widths
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        for i in range(args.n_hidden_layers+2):
            net.net[i].subnet_widths=args.widths
            net.net[i].subnet_index=len(args.widths)-1
        
        net.eval()
        for e in trange(1, args.epochs+1):
            optimizer.zero_grad()
            loss_total = 0
            for w_idx, width in enumerate(args.widths):
                if args.experiment in ['image_spatial', 'video_temporal']:
                    y=dataset.mask(input=dataset.y, w_idx=w_idx, w_num=len(args.widths))
                net.select_subnet(w_idx)
                yhat=net(x)
                loss=loss_fn(yhat, y)
                loss_total+=loss.item()
                loss.backward()
            optimizer.step()
            if e%args.log_iter==0:
                print(f"EPOCH: {e}/{args.epochs} AVG LOSS: {loss_total/len(args.widths)}")

        print("training done")

        with torch.no_grad():
            torch.save(net.state_dict(), os.path.join(result_dir, 'model.pth'))
            for w_idx, width in enumerate(args.widths):
                net.select_subnet(w_idx)
                evaluate(net, w_idx)
            evaluate.metrics(n=len(net.runnable_widths))


    def train_individual(self, args, device, dataset, result_dir, loss_fn, evaluate):
        '''
        <predefined widths used in the paper>
        image spectral growing: 30, 60, 90, 120
        image spatial growing: 59, 59, 59, 59
        '''
        x = dataset.x
        y = dataset.y
        print("training individual model")

        for w_idx, width in enumerate(args.widths):
            print(f'current width: {width}')
            net=ProgressiveSiren(
                in_feats=x.shape[-1],
                hidden_feats=width,
                n_hidden_layers=args.n_hidden_layers,
                out_feats=y.shape[-1],
                device=device).to(device)
            print(f'number of parameters: {sum(p.numel() for p in net.parameters())}')

            if args.experiment in ['image_spatial', 'video_temporal']:
                y=dataset.mask(input=dataset.y, w_idx=w_idx, w_num=len(args.widths))

            optimizer=torch.optim.Adam(net.parameters(), lr=args.lr)
            net.train()
            net.select_subnet(0)
            for e in trange(args.epochs):
                optimizer.zero_grad()
                yhat=net(x)
                loss=loss_fn(yhat, y)
                loss.backward()
                optimizer.step()

                if e % args.log_iter == 0:
                    print(f"EPOCH: {e}/{args.epochs} WIDTH: {width} LOSS: {loss.item()}")

            with torch.no_grad():
                torch.save(net.state_dict(), os.path.join(result_dir, f'model_{w_idx}.pth'))
                net.eval()
                evaluate(net, w_idx)
        
        print("training done")
        evaluate.metrics(n=len(args.widths))
