import argparse
import warnings

warnings.filterwarnings("ignore")

import torch, pickle
from torch.optim import Adam
from model import ARGA_NA, GCNEncoder, Discriminator
from torch_geometric.utils import degree

from utils import load_data, fix_seed, evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CiteSeer',
                    choices=['Cora', 'CiteSeer', 'PubMed', 'CoraFull'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--emb_dim', type=int, default=32)
parser.add_argument('--threshold', type=int, default=3)
parser.add_argument('--val_ratio', type=float, default=0.05)
parser.add_argument('--test_ratio', type=float, default=0.15)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--discriminator_lr', type=float, default=0.0001)

args = parser.parse_args()

dataset = args.dataset
epochs = args.epochs
d_t = args.threshold
lr = args.learning_rate
discriminator_lr = args.discriminator_lr
hid_dim, out_dim = 2 * args.emb_dim, args.emb_dim
val_ratio, test_ratio = args.val_ratio, args.test_ratio
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset and split links
try:
    with open(f'./data/{dataset}_{val_ratio}_{test_ratio}.pkl', 'rb') as f:
        data = pickle.load(f)
except:
    fix_seed(0)  # fix random split
    data = load_data(dataset, val_ratio, test_ratio)
    with open(f'./data/{dataset}_{val_ratio}_{test_ratio}.pkl', 'wb') as f:
        pickle.dump(data, f)

x, edge_index = data.x, data.train_pos_edge_index
row, col = edge_index
mask = row < col
row, col = row[mask], col[mask]
train_pos_edge_index = torch.stack([row, col], dim=0)

D = degree(edge_index[0], x.shape[0])


def get_self_loops(D, d_t):
    self_loop_index = []
    num_per_loop = []
    for d in range(d_t):
        loop_index = torch.nonzero(D == d).t()
        loop_index = loop_index.repeat(2, 1)
        self_loop_index.append(loop_index)
        num_per_loop.append(torch.full((loop_index.shape[1],), d_t - d))
    return torch.cat(self_loop_index, dim=1), torch.cat(num_per_loop)


self_loop_index, num_per_loop = get_self_loops(D, d_t=d_t)
num_per_loop = num_per_loop.to(device)
x, edge_index = x.to(device), edge_index.to(device)

encoder = GCNEncoder(x.shape[1], hid_dim, out_dim)
discriminator = Discriminator(out_dim, hid_dim, out_dim)
model = ARGA_NA(encoder, discriminator).to(device)
encoder_optimizer = Adam(encoder.parameters(), lr=lr)
discriminator_optimizer = Adam(discriminator.parameters(),
                                               lr=discriminator_lr)

for epoch in range(epochs):
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(x, edge_index)

    for i in range(5):
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()

    loss = model.recon_loss(z, train_pos_edge_index, self_loop_index, num_per_loop)
    loss = loss + model.reg_loss(z)

    print(f'Epoch: {epoch:03d}, Loss: {float(loss):.4f}')

    loss.backward()
    encoder_optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        auc, ap, hitsK = evaluate(z, model.decoder,
                                  data.val_pos_edge_index, data.val_neg_edge_index)
        print(f'  V auc={auc:.4f} ap={ap:.4f} hitsK={hitsK:.4f}')

        auc, ap, hitsK = evaluate(z, model.decoder,
                                  data.test_pos_edge_index, data.test_neg_edge_index)
        print(f'  T auc={auc:.4f} ap={ap:.4f} hitsK={hitsK:.4f}')
