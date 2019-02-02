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
parser.add_argument("--dataset", type=str, default="cora",
                    help="Dataset string ('aifb', 'mutag', 'bgs', 'am', 'cora')")

parser.add_argument("--nEpochs", type=int, default=10,
                    help="Learning rate of the optimizer")
parser.add_argument("--eval_every", type=int, default=25,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=50,
                    help="Interval of epochs to save a checkpoint of the model?")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")

parser.add_argument("--optimizer", type=str, default="Adam",
                    help="Which optimizer to use?")
parser.add_argument("--lr", type=float, default=0.1,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")
parser.add_argument("--clip", type=int, default=1000,
                    help="Maximum gradient norm allowed.")
parser.add_argument("--margin", type=int, default=1,
                    help="The margin between positive and negative samples in the max-margin loss")

parser.add_argument("--emb_dim", type=int, default=16,
                    help="Entity embedding size")
parser.add_argument("--feat_in", type=int, default=1433,
                    help="Input feature size")
parser.add_argument("--gcn_layers", type=int, default=1,
                    help="Number of GCN layers")
parser.add_argument("--n_class", type=int, default=7,
                    help="Number of classes in classification task")

parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Run the code in debug mode?")
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

gcn, _, sm_classifier = initialize_model(params)

trainer = Trainer(params, gcn, None, sm_classifier, classifier_data, None)
evaluator = Evaluator(params, gcn, None, sm_classifier, classifier_data, None, None)

logging.info('Starting training with full batch...')

# tb_logger = Logger(params.exp_dir)

for e in range(params.nEpochs):
    tic = time.time()
    loss = trainer.classifier_one_step()
    toc = time.time()

    # tb_logger.scalar_summary('loss', loss, e)

    logging.info('Epoch %d with loss: %f in %f'
                 % (e, loss, toc - tic))

    if (e + 1) % params.eval_every == 0:
        log_data = evaluator.classifier_log_data()
        logging.info('Performance:' + str(log_data))

        # for tag, value in log_data.items():
        # tb_logger.scalar_summary(tag, value, e + 1)

        to_continue = trainer.save_classifier(log_data)
        if not to_continue:
            break

    if (e + 1) % params.save_every == 0:
        torch.save(gcn, os.path.join(params.exp_dir, 'gcn_checkpoint.pth'))
        torch.save(sm_classifier, os.path.join(params.exp_dir, 'sm_checkpoint.pth'))
