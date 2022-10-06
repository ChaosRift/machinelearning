import sys
sys.path.insert(0, 'src')
import transform, numpy as np, os
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import numpy
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

# this helps with memory issues. I need to go full TF 2.x for it to work. 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_GPU_ALLOCATOR'] = ''
os.environ['CUDA_VISIBLE_DEVICES']="0" 
tf.config.list_physical_devices()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BATCH_SIZE = 1
DEVICE = '/gpu:0'
#DEVICE = '/cpu:0'

####### Workaround: downgrading imageio-ffmpeg version from 0.4.3 to 0.2.0 resolves this issue.
# pip install imageio-ffmpeg==0.2.0
def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=1):
    video_clip = VideoFileClip(path_in, audio=False)

    # For outputting video it is imperative to produce the highest quality for post processing. 
    # I found that if this is the last step the quality goes to heck. 
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="veryslow", bitrate="90000k",    
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with  tf.compat.v1.Session(config=config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #print("Opening checkpoint: ", ckpt.model_checkpoint_path )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            #print("count", count, "leng", len(_preds))
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))
                
        total_frames = 0
        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            total_frames +=1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0
                print("frames processed:", total_frames)
        if frame_count != 0:
            style_and_write(frame_count)
        
            style_and_write(frame_count)
        print("FInishing up video. Total frames:", total_frames)
        video_clip.close()            
        video_writer.close()


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=2):
    #print(f"Opening image {data_in}")
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        #img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    #with tf.compat.v1.Session(config=config) as sess:
    with g.as_default(), g.device(device_t), tf.compat.v1.Session(config=config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            #print("Opening checkpoint: ", ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            #print("Opening checkpoint from dir: ", checkpoint_dir)
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                print(f"Current Batch Image {curr_batch_in}")
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, 'Images have different dimensions. ' + 'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])

        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    # call again if more images.
    print("SESSION IS: ", type(sess) )
    del preds
    sess.close()
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, device_t=device_t, batch_size=1)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/gpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_preprocess(in_path, out_path, checkpoint_dir, device_t=DEVICE, batch_size=2):
    print(f"Building shape database, checkpoint {checkpoint_dir}")
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    shapes = 0
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        if shape not in in_path_of_shape:
            shapes += 1
            print(f'Shapes Processed {shapes}')
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
        
    print("Done...")
    
    for shape in in_path_of_shape:
        print(f"Processing images of size {shape}")
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], checkpoint_dir, device_t, batch_size)
        time.sleep(2)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',
                        help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.out_path):
        os.mkdir(opts.out_path)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path
        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir, device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        ffwd_preprocess(full_in, full_out, opts.checkpoint_dir, device_t=opts.device, batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
