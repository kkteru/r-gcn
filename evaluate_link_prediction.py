import argparse
import logging
import pdb

from core import *
from managers import *
from utils import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="The best modeel saved in this folder would be loaded")
parser.add_argument("--gpu", type=int, default=0,
                    help="Which GPU to use?")

parser.add_argument("--no_encoder", action='store_true',
                    help="Run the code in debug mode?")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument("--neg_sample_size", type=int, default=30,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument('--filter', action='store_true',
                    help='Filter the samples while evaluation')
parser.add_argument('--eval_mode', type=str, default="head",
                    help='Evaluate on head and/or tail prediction?')
parser.add_argument('--dataset', type=str, default="full",
                    help='Which dataset to use? "full" for FB15K, "mod" for FB15K237')

params = parser.parse_args()

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

if params.dataset == 'mod':
    params.total_rel = 237
    params.total_ent = 14541
    DATA_PATH = os.path.join(MAIN_DIR, 'data/FB15K237')
else:
    params.total_rel = 1345
    params.total_ent = 14951
    DATA_PATH = os.path.join(MAIN_DIR, 'data/FB15K')

TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train2id.txt')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid2id.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test2id.txt')
ALL_DATA_PATH = os.path.join(DATA_PATH, 'triple2id.txt')

params.total_rel = 1345
params.total_ent = 14951

exps_dir = os.path.join(MAIN_DIR, 'experiments')
params.exp_dir = os.path.join(exps_dir, params.experiment_name)

test_data_sampler = DataSampler(params, TEST_DATA_PATH, ALL_DATA_PATH)
gcn, distmul = initialize_model(params)
evaluator = Evaluator(params, gcn, distmul, test_data_sampler)

logging.info('Testing models from %s' % os.path.join(params.exp_dir))

log_data = evaluator.get_log_data(params.eval_mode)
logging.info('Test performance:' + str(log_data))
