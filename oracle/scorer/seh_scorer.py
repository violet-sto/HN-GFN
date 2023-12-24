import pickle
import torch
import gzip
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from generator.gfn import make_model
from rdkit import Chem
import numpy as np

models = {}
bpath = "./data/blocks_105.json"
proxy_path = "oracle/scorer/seh"

class seh_model:
    def __init__(self, bpath, device):
        eargs = pickle.load(gzip.open(f'{proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.mdp.floatX = torch.float32  # torch.dtype
        eargs.device = device  # torch.device
        eargs.floatX = 'float32'
        self.proxy = make_model(eargs, self.mdp)
        super_hackish_param_map = {
            'mpnn.lin0.weight': params[0],
            'mpnn.lin0.bias': params[1],
            'mpnn.conv.bias': params[3],
            'mpnn.conv.nn.0.weight': params[4],
            'mpnn.conv.nn.0.bias': params[5],
            'mpnn.conv.nn.2.weight': params[6],
            'mpnn.conv.nn.2.bias': params[7],
            'mpnn.conv.lin.weight': params[2],
            'mpnn.gru.weight_ih_l0': params[8],
            'mpnn.gru.weight_hh_l0': params[9],
            'mpnn.gru.bias_ih_l0': params[10],
            'mpnn.gru.bias_hh_l0': params[11],
            'mpnn.lin1.weight': params[12],
            'mpnn.lin1.bias': params[13],
            'mpnn.lin2.weight': params[14],
            'mpnn.lin2.bias': params[15],
            'mpnn.set2set.lstm.weight_ih_l0': params[16],
            'mpnn.set2set.lstm.weight_hh_l0': params[17],
            'mpnn.set2set.lstm.bias_ih_l0': params[18],
            'mpnn.set2set.lstm.bias_hh_l0': params[19],
            'mpnn.lin3.weight': params[20],
            'mpnn.lin3.bias': params[21],
        }
        for k, v in super_hackish_param_map.items():
            self.proxy.get_parameter(k).data = torch.tensor(v, dtype=self.mdp.floatX)
        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()


def get_scores(task, mols, device=None):
    model = models.get(task)
    if model is None:
        model = seh_model(bpath, device)
        models[task] = model

    if mols is not None:
        return np.clip(model(mols),0.,10.) / 10.
    else:
        return []