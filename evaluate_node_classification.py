import argparse
import logging
import time
import pickle as pkl

from core import *
from managers import *
from utils import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")
parser.add_argument("--dataset", type=str, default="aifb",
                    help="Dataset string ('aifb', 'mutag', 'bgs', 'am', 'cora')")

parser.add_argument("--emb_dim", type=int, default=16,
                    help="Entity embedding size")
parser.add_argument("--gcn_layers", type=int, default=1,
                    help="Number of GCN layers")

parser.add_argument("--no_encoder", type=bool_flag, default=False,
                    help="Run the code in debug mode?")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

params = parser.parse_args()

initialize_experiment(params)

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

with open(MAIN_DIR + '/' + params.dataset + '.pickle', 'rb') as f:
    classifier_data = pkl.load(f)

classifier_data['A'] = list(map(get_torch_sparse_matrix, classifier_data['A'], [params.device] * len(classifier_data['A'])))

params.total_rel = len(classifier_data['A'])
params.total_ent = classifier_data['A'][0].shape[0]

logging.info('Loaded %s dataset with %d entities and %d relations' % (params.dataset, params.total_ent, params.total_rel))

gcn, sm_classifier = initialize_model(params, classifier_data)

evaluator = Evaluator(params, gcn, sm_classifier, classifier_data)

logging.info('Testing model %s' % os.path.join(params.exp_dir, 'best_model.pth'))

log_data = evaluator.classifier_log_data(data='test')
logging.info('Test performance:' + str(log_data))
