from argparse import ArgumentParser
import sys
sys.path.insert(0, 'src')
import random
import evaluate
from utils import exists, list_files

checkpointDir = "checkpoints"
TMP_DIR = '.fns_frames_%s/' % random.randint(0, 99999)
DEVICE = '/gpu:0'
BATCH_SIZE = 4

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style', type=str,
                        dest='style', help='Style name which loads checkpoint',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='in video path',
                        metavar='IN_PATH', required=True)
    
    parser.add_argument('--tmp-dir', type=str, dest='tmp_dir',
                        help='tmp dir for processing', metavar='TMP_DIR',
                        default=TMP_DIR)

    parser.add_argument('--device', type=str, dest='device',
                        help='device for eval. CPU discouraged. ex: \'/gpu:0\'',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for eval. default 4.',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--no-disk', type=bool, dest='no_disk',
                        help='Don\'t save intermediate files to disk. Default False',
                        metavar='NO_DISK', default=True)
    return parser

def check_opts(opts):
    exists(opts.checkpoint)

def main():
    parser = build_parser()
    opts = parser.parse_args()
    style = opts.style
    checkpoint = f"{checkpointDir}/{style}/"
    movieName = opts.in_path.split(".")[:-1]
    outName = "_".join(movieName) + f"_{style}.mp4"
    evaluate.ffwd_video(opts.in_path, outName, checkpoint, opts.device, opts.batch_size)

 
if __name__ == '__main__':
    main()


