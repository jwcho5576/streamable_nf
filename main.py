import argparse

from dataset import *
from network import ProgressiveSiren
from train.image_trainer import ImageTrainer
from train.sdf_trainer import SDFTrainer
from train.video_trainer import VideoTrainer
from utils.evaluation import *
from utils.metrics import MSE
from utils.utils import sdf_loss

if __name__=='__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # experiment settings
    parser.add_argument('--experiment', type=str, default='image_spectral', help='image_spectral/image_spatial/sdf_spectral/video_temporal')
    parser.add_argument('--model', type=str, default='progressive', help='progressive/slimmable/individual')

    # data settings
    parser.add_argument('--kodak_num', type=int, default=1, help='select kodak image (1~24)')
    parser.add_argument('--shape_data', type=str, default='dragon', help='dragon/armadillo/happy_buddha')
    parser.add_argument('--uvg_data', type=str, default='ReadySetGo', help='Beauty/Bosphorus/HoneyBee/Jockey/ReadySetGo/ShakeNDry/YachtRide')

    # network settings
    parser.add_argument('--widths', nargs='+', type=int, required=True, help='runnable network widths')
    parser.add_argument('--n_hidden_layers', type=int, default=4, help='number of hidden layers')
    
    # optimizer settings
    parser.add_argument('--epochs', type=int, default=50000, help='training step')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

    # log settings
    parser.add_argument('--root_dir', type=str, default=None, help='root directory')
    parser.add_argument('--log_iter', type=int, default=100, help='print log every...')

    # etc.
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--frame_batchsize', type=int, default=1, help='number of frames to process in parallel')
    parser.add_argument('--pointcloud_batchsize', type=int, default=131072, help='batch size for point cloud')

    args = parser.parse_args()


    # randomness control
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # set data and directory
    if args.root_dir == None:
        args.root_dir = os.getcwd()

    if args.experiment in ['image_spectral', 'image_spatial']:
        dataset = Kodak(args=args, num=args.kodak_num, device=device)
        result_dir = os.path.join(
            args.root_dir,
            'results',
            args.experiment,
            f'kodim{args.kodak_num:02}',
            args.model)
        trainer = ImageTrainer()
        loss_fn = MSE()
        evaluate = ImageEval(result_dir, dataset, args)

    elif args.experiment == 'video_temporal':
        dataset = UVG(args=args, name=args.uvg_data, device=device)
        result_dir = os.path.join(
            args.root_dir,
            'results',
            args.experiment,
            args.uvg_data,
            args.model)
        loss_fn = MSE()
        evaluate = VideoEval(result_dir, dataset, args)
        trainer = VideoTrainer()

    elif args.experiment == 'sdf_spectral':
        dataset = PointCloud(f'./data/shape/{args.shape_data}.xyz', args.pointcloud_batchsize)
        result_dir = os.path.join(
            args.root_dir,
            'results',
            args.experiment,
            args.shape_data,
            args.model)
        loss_fn = sdf_loss
        evaluate = SDFEval(result_dir, dataset, args)
        trainer = SDFTrainer()

    else:
        raise NotImplementedError

    os.makedirs(result_dir, exist_ok=True)
    
    # build network and train
    if args.model == 'progressive':
        # build with smallest width
        net = ProgressiveSiren(
            in_feats=dataset.x.shape[-1],
            hidden_feats=args.widths[0],
            n_hidden_layers=args.n_hidden_layers,
            out_feats = dataset.y.shape[-1],
            device=device).to(device)
        trainer.train_progressive(
            args=args,
            net=net,
            dataset=dataset,
            result_dir=result_dir,
            loss_fn=loss_fn,
            evaluate=evaluate)

    elif args.model == 'slimmable':
        # build with largest width
        net = ProgressiveSiren(
            in_feats=dataset.x.shape[-1],
            hidden_feats=args.widths[-1],
            n_hidden_layers=args.n_hidden_layers,
            out_feats = dataset.y.shape[-1],
            device=device).to(device)
        trainer.train_slimmable(
            args=args,
            net=net,
            dataset=dataset,
            result_dir=result_dir,
            loss_fn=loss_fn,
            evaluate=evaluate)

    elif args.model == 'individual':
        # network will be built inside the trainer
        trainer.train_individual(
            args=args,
            device=device,
            dataset=dataset,
            result_dir=result_dir,
            loss_fn=loss_fn,
            evaluate=evaluate)

    else:
        raise NotImplementedError
