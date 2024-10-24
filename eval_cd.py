from argparse import ArgumentParser
import torch
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--log_dir', default='checkpoints', type=str)


    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    parser.add_argument('--input_channels', type=int, default=3, help='image input channels')
    parser.add_argument('--in_channels', type=int, default=256, help='in channels of transformer feature')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Transformer feature dimension')
    parser.add_argument('--num_queries', type=int, default=100, help='number of queries')
    parser.add_argument('--nhead', type=int, default=8, help='number of heads')
    parser.add_argument('--drop_out', type=int, default=0, help='dropout in Transformer')
    parser.add_argument('--dim_forward', type=int, default=2048, help='feature dimension in feedforward network')
    parser.add_argument('--enc_layers', type=int, default=0, help=' number of Transformer encoder layers')
    parser.add_argument('--dec_layers', type=int, default=6, help=' number of Transformer decoder layers')
    parser.add_argument('--mask_dim', type=int, default=256, help=' mask feature dimension')
    parser.add_argument('--enforce_input_project', type=bool, default=True, help=' mask feature dimension')
    parser.add_argument('--rate', type=float, default=0.5)
    parser.add_argument('--area_the', type=int, default=0)
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--pretrain', action='store_true', help='use gwc pretrain or not')
    parser.add_argument('--pretrain_path', type=str, default="./pretrain", help='use gwc pretrain or not')
    parser.add_argument('--vis_query', action='store_true', help='visualize query or not')

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    args.logdir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    test(args)
