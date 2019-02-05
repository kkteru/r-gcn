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
parser.add_argument("--no_encoder", type=bool_flag, default=False,
                    help="Run the code in debug mode?")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

params = parser.parse_args()

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

exps_dir = os.path.join(MAIN_DIR, 'experiments')
params.exp_dir = os.path.join(exps_dir, params.experiment_name)

test_data_sampler = DataSampler(params, TEST_DATA_PATH)
gcn, distmul, _ = initialize_model(params)
evaluator = Evaluator(params, gcn, distmul, None, None, test_data_sampler, 30)

logging.info('Testing model %s' % os.path.join(params.exp_dir, 'best_model.pth'))

log_data = evaluator.link_log_data()
logging.info('Test performance:' + str(log_data))
