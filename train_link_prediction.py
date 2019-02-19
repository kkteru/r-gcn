import argparse
import logging
import time

from core import *
from managers import *
from utils import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")

parser.add_argument("--nEpochs", type=int, default=10,
                    help="Learning rate of the optimizer")
parser.add_argument("--nBatches", type=int, default=200,
                    help="Batch size")
parser.add_argument("--eval_every", type=int, default=25,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=50,
                    help="Interval of epochs to save a checkpoint of the model?")
parser.add_argument('--eval_mode', type=str, default="head",
                    help='Evaluate on head and/or tail prediction?')
parser.add_argument('--filter', action='store_true',
                    help='Filter the samples while evaluation')

parser.add_argument("--neg_sample_size", type=int, default=30,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--optimizer", type=str, default="Adam",
                    help="Which optimizer to use?")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")
parser.add_argument("--clip", type=int, default=1000,
                    help="Interval of epochs to save a checkpoint of the model?")
parser.add_argument("--margin", type=int, default=1,
                    help="The margin between positive and negative samples in the max-margin loss")

parser.add_argument("--emb_dim", type=int, default=50,
                    help="Entity embedding size")
parser.add_argument("--feat_in", type=int, default=14951,
                    help="Entity embedding size")
parser.add_argument("--gcn_layers", type=int, default=1,
                    help="Number of GCN layers")
parser.add_argument("--n_class", type=int, default=4,
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

params.total_rel = 1345
params.total_ent = 14951

link_train_data_sampler = DataSampler(params, TRAIN_DATA_PATH, ALL_DATA_PATH, params.nBatches, params.debug)
link_valid_data_sampler = DataSampler(params, VALID_DATA_PATH, ALL_DATA_PATH,)

gcn, distmult = initialize_model(params)

trainer = Trainer(params, gcn, distmult, link_train_data_sampler)
evaluator = Evaluator(params, gcn, distmult, link_valid_data_sampler)

batch_size = int(len(link_train_data_sampler.data) / params.nBatches)

logging.info('Starting training with batch size %d' % batch_size)

# tb_logger = Logger(params.exp_dir)

for e in range(params.nEpochs):
    res = 0
    tic = time.time()
    for b in range(params.nBatches):
        loss = trainer.link_pred_one_step(b)
        res += loss
    toc = time.time()

    # tb_logger.scalar_summary('loss', loss, e)

    logging.info('Epoch %d with loss: %f and emb norm %f in %f'
                 % (e, res, torch.mean(trainer.encoder.ent_emb), toc - tic))

    if (e + 1) % params.eval_every == 0:
        tic = time.time()
        log_data = evaluator.get_log_data(params.eval_mode)
        toc = time.time()
        logging.info('Performance: %s in %f' % (str(log_data), (toc - tic)))

        # for tag, value in log_data.items():
        # tb_logger.scalar_summary(tag, value, e + 1)

        to_continue = trainer.save_link_predictor(log_data)
        if not to_continue:
            break
    if (e + 1) % params.save_every == 0:
        torch.save(gcn, os.path.join(params.exp_dir, 'gcn_checkpoint.pth'))
        torch.save(distmult, os.path.join(params.exp_dir, 'distmult_checkpoint.pth'))
