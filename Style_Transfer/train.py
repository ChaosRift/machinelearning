import sys, os
sys.path.insert(0, 'src')
from optimize import optimize
from argparse import ArgumentParser
from utils import get_img, exists, get_file_list
import evaluate
import glob, random

# I suck at SN
# https://coolconversion.com/math/scientific-notation-to-decimal/
CONTENT_WEIGHT = 7.5e0
TV_WEIGHT = 2e2         # 200
#STYLE_WEIGHT = 5e1      # 50
STYLE_WEIGHT = 1e2      # 100 -- Default
#STYLE_WEIGHT = 2e2      # 200
#STYLE_WEIGHT = 5e2      # 500


#LEARNING_RATE = 1e-2    # 0.01
LEARNING_RATE = 1e-3    # 0.001   --- Default
#LEARNING_RATE = 1e-4    # 0.0001

NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 1000    # When to print stats and save model.
VGG_PATH = 'F:/machine_learning/pretrained_models/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'F:/machine_learning/datasets/train_full'
BATCH_SIZE = 5  # Training data upscaled to 2048 and then read at 512... very large. 
DEVICE = '/gpu:0'
FRAC_GPU = 1 # Supposed to be for GPU memory tuning. 

# I added these, removed these may use them again. 
TEST_IN = 'test'
TEST_OUT = 'test_out'
TEST_IMG = 'test.jpg'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    # can be deleted, but I kept out of laziness for the below asserts. 
    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    return parser

def check_opts(opts):
    if not os.path.isdir(opts.checkpoint_dir):
        os.mkdir(opts.checkpoint_dir)
    print(opts.style)
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, f"style path ({opts.style}) not found!")
    exists(opts.train_path, "train path not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def _get_files(img_dir):
    # For large datasets, files may be split into smaller subdirs. 
    # This gets everything and maintains the full path.
    files = get_file_list(img_dir)
    return files

    
def main():
    lastimg = None
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)
    if len(style_target) < 1:
        print("Image load failed")
        style_target = lastimg
    else:
        lastimg = style_target

    content_targets = _get_files(options.train_path)

    kwargs = {
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
        "learning_rate":options.learning_rate
    }

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    styleName = options.style.split("\\")[-1].split('.')[0]
    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses
        print("\n************************************************************************")
        print(f"                              Epoch {epoch}                              ")
        print(f"                      Iteration: {i}, Loss: {loss}                       ")
        print(f"         style: {style_loss}, content: {content_loss}, tv: {tv_loss}     ")
        print("************************************************************************")
        
        testFiles = glob.glob("test/*.jpg")
        preds_path = f'test_out/{styleName}_{epoch}_{i}.jpg'
        ckpt_dir = os.path.dirname(options.checkpoint_dir)
        evaluate.ffwd_to_img(random.choice(testFiles),preds_path, options.checkpoint_dir)

    ckpt_dir = options.checkpoint_dir
    cmd_text = f'python evaluate.py --checkpoint {ckpt_dir} ...'
    print(f"Training complete. For evaluation:\n`{cmd_text}`")

if __name__ == '__main__':
    main()