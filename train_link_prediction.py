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
                    help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

parser.add_argument("--nEpochs", type=int, default=10,
                    help="Learning rate of the optimizer")
parser.add_argument("--nBatches", type=int, default=200,
                    help="Batch size")
parser.add_argument("--eval_every", type=int, default=25,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=50,
                    help="Interval of epochs to save a checkpoint of the model?")

parser.add_argument("--sample_size", type=int, default=30,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use?")
parser.add_argument("--lr", type=float, default=0.1,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")
parser.add_argument("--clip", type=int, default=1000,
                    help="Interval of epochs to save a checkpoint of the model?")

parser.add_argument("--emb_dim", type=int, default=50,
                    help="Entity embedding size")
parser.add_argument("--gcn_layers", type=int, default=2,
                    help="Number of GCN layers")
parser.add_argument("--n_class", type=int, default=4,
                    help="Number of classes in classification task")

parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Run the code in debug mode?")
parser.add_argument("--no_encoder", type=bool_flag, default=True,
                    help="Run the code in debug mode?")

params = parser.parse_args()

initialize_experiment(params)

link_train_data_sampler = DataSampler(TRAIN_DATA_PATH, params.debug)
link_valid_data_sampler = DataSampler(VALID_DATA_PATH)

params.total_rel = 475
params.total_ent = 14541

logging.info('Loaded %s dataset with %d entities and %d relations' % (params.dataset, params.total_ent, params.total_rel))

gcn, distmul, _ = initialize_model(params)

# print([model_params.data.shape for model_params in gcn.parameters()])
# print([model_params.data.shape for model_params in distmul.parameters()])
# print([model_params.data.shape for model_params in sm_classifier.parameters()])

# for m_param in gcn.parameters():
#     print(type(m_param.data), m_param.size())

trainer = Trainer(params, gcn, distmul, None, None, link_train_data_sampler)
evaluator = Evaluator(gcn, distmul, None, None, link_valid_data_sampler, params.sample_size)

batch_size = int(len(link_train_data_sampler.data) / params.nBatches)

logging.info('Starting training with batch size %d' % batch_size)

tb_logger = Logger(params.exp_dir)

for e in range(params.nEpochs):
    tic = time.time()
    for b in range(params.nBatches):
        loss = trainer.link_pred_one_step(batch_size)
    toc = time.time()

    tb_logger.scalar_summary('loss', loss, e)

    torch.mean(link_pred_one_step.encoder.ent_emb)

    logging.info('Epoch %d with loss: %f and emb norm %f in %f'
                 % (e, loss, torch.mean(link_pred_one_step.encoder.ent_emb), toc - tic))
    if (e + 1) % params.eval_every == 0:
        log_data = evaluator.link_log_data()
        logging.info('Performance:' + str(log_data))

        for tag, value in log_data.items():
            tb_logger.scalar_summary(tag, value, e + 1)

        to_continue = trainer.save_link_predictor(log_data)
        if not to_continue:
            break
    if (e + 1) % params.save_every == 0:
        torch.save(gcn, os.path.join(params.exp_dir, 'gcn_checkpoint.pth'))
        torch.save(distmul, os.path.join(params.exp_dir, 'distmul_checkpoint.pth'))
