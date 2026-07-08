import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--inp_seq', default=7, type=int, help='time step of input data')
    parser.add_argument('--tar_seq', default=1, type=int, help='time step of target data')
    parser.add_argument('--inp_features', default=7, type=int, help='the number of the features of input data')
    parser.add_argument('--data_idx', default=[0, 1, 2, 3, 4, 5, 6], type=list, 
                        help="index of used data ['Vp', 'Alpha_ratio', 'entropy', 'Alfvenicity', 'Np', 'Tp', 'coulomb num']")

    ### model parameters 
    parser.add_argument('--res_dir', default='./results', type=str, help='the directory to save the results')
    parser.add_argument('--ex_name', default='ver_260106_tar3', type=str, help='the name of the experiment')
    parser.add_argument('--model_name', default='TSMIXER', type=str, help='the name of the model')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size for fcn layers')
    parser.add_argument('--n_blocks', default=4, type=int, help='the number of blocks of tsmixer model')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')

    ### training parameters
    parser.add_argument('--use_gpu', default=True, type=bool, help='whether use gpu')
    parser.add_argument('--gpu', default='3', type=str, help='the gpu id')
    parser.add_argument('--seed', default=777, type=int, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='validation batch size')
    parser.add_argument('--save_model', default=False, type=bool, help='whether save model')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')

    ## data mode
    parser.add_argument('--flag', default='train', type=str, help='whether train or test')
    parser.add_argument('--mode', default='full', type=str, help='whether slow or fast')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    args.flag = 'train'
    Exp(args).model_train()
    
    args.flag = 'test'
    Exp(args).model_test()
    print(f"End of Training and test process - ex:{args.ex_name}")
        
        
    # columns = ['Vp', 'Alpha_ratio', 'entropy', 'Alfvenicity', 'Np', 'Tp', 'coulomb num']
    # idx_filter = [None]
    
    # for i in idx_filter:
    #     if i is None:
    #         args.data_idx = list(range(len(columns)))
    #         args.inp_features = len(args.data_idx)
        
    #     else:
    #         args.data_idx = list(range(len(columns)))
    #         args.data_idx.remove(i)
    #         args.inp_featues = len(args.data_idx)
    #         args.ex_name = f"ver_251203_2_no_{columns[i]}"
    #     print(f"Start of Training and test process - ex:{args.ex_name}, used idx: {args.data_idx}")        
    #     args.flag = 'train'
    #     Exp(args).model_train()
        
    #     args.flag = 'test'
    #     Exp(args).model_test()
    #     print(f"End of Training and test process - ex:{args.ex_name}")
        
