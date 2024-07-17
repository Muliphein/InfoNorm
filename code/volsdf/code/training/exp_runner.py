import sys

sys.path.append('../code')
import argparse
import GPUtil
import torch
from training.volsdf_train import VolSDFTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=150000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument('--scan_id', type=str, default=-1, required=True)
    parser.add_argument('--split', type=str, required=True)

    opt = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

    trainrunner = VolSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    split=opt.split,
                                    )

    trainrunner.run()
