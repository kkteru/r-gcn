import os
import argparse
import logging
import json
import torch
import pdb

# import tensorflow as tf
from core import DistMult, GCN, EmbLookUp

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(MAIN_DIR, 'data/FB15K')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train2id.txt')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid2id.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test2id.txt')
ALL_DATA_PATH = os.path.join(DATA_PATH, 'triple2id.txt')


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def initialize_experiment(params):

    exps_dir = os.path.join(MAIN_DIR, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params):

    if os.path.exists(os.path.join(params.exp_dir, 'best_enc.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_enc.pth'))
        enc = torch.load(os.path.join(params.exp_dir, 'best_enc.pth'))  # Update these
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_dec.pth'))
        dec = torch.load(os.path.join(params.exp_dir, 'best_dec.pth'))  # Update these
    else:
        logging.info('No existing model found. Initializing new model..')
        if params.no_encoder:
            enc = EmbLookUp(params, params.feat_in).to(device=params.device)
        else:
            enc = GCN(params, params.feat_in).to(device=params.device)
        dec = DistMult(params).to(device=params.device)

    return enc, dec


# class Logger(object):

#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = tf.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)

#     def image_summary(self, tag, images, step):
#         """Log a list of images."""

#         img_summaries = []
#         for i, img in enumerate(images):
#             s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")

#             # Create an Image object
#             img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                        height=img.shape[0],
#                                        width=img.shape[1])
#             # Create a Summary value
#             img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)

#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""

#         # Create a histogram using numpy
#         counts, bin_edges = np.histogram(values, bins=bins)

#         # Fill the fields of the histogram proto
#         hist = tf.HistogramProto()
#         hist.min = float(np.min(values))
#         hist.max = float(np.max(values))
#         hist.num = int(np.prod(values.shape))
#         hist.sum = float(np.sum(values))
#         hist.sum_squares = float(np.sum(values**2))

#         # Drop the start of the first bin
#         bin_edges = bin_edges[1:]

#         # Add bin edges and counts
#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)

#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()
