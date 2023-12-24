# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
# copied from: https://github.com/bytedance/markov-molecular-sampling/blob/master/estimator/scorer/scorer.py

import math
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx

from utils.chem import standardize_smiles
from . import sa_scorer, kinase_scorer, drd2_scorer, seh_scorer
# from . import chemprop_scorer

### get scores
def get_scores(objective, mols, device=None):
    if objective == 'seh':
        scores = [seh_scorer.get_scores(objective, mol, device=device) \
            for mol in mols] 
    else:
        mols = [m.mol for m in mols]
        mols = [standardize_smiles(mol) for mol in mols]
        mols_valid = [mol for mol in mols if mol is not None]
        
        if objective == 'drd2':
            scores = drd2_scorer.get_scores(mols_valid)
        elif objective == 'jnk3' or objective == 'gsk3b':
            scores = kinase_scorer.get_scores(objective, mols_valid)
        elif objective.startswith('chemprop'):
            scores = chemprop_scorer.get_scores(objective, mols_valid, device=device)

        else: scores = [get_score(objective, mol) for mol in mols_valid]
        
    scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
    return scores

def get_score(objective, mol):
    try:
        if objective == 'qed': 
            return QED.qed(mol)
        elif objective == 'sa': 
            x = sa_scorer.calculateScore(mol)
            return (10. - x) / 9. # normalized to [0, 1]
        elif objective == 'mw': # molecular weight
            return mw(mol)
        elif objective == 'logp': # real number
            return Descriptors.MolLogP(mol)
        elif objective == 'penalized_logp':
            return penalized_logp(mol)
        elif 'rand' in objective:
            raise NotImplementedError
            # return rand_scorer.get_score(objective, mol)
        else: raise NotImplementedError
    except ValueError:
        return 0.

    
### molecular properties
def mw(mol):
    '''
    molecular weight estimation from qed
    '''
    x = Descriptors.MolWt(mol)
    a, b, c, d, e, f = 2.817, 392.575, 290.749, 2.420, 49.223, 65.371
    g = math.exp(-(x - c + d/2) / e)
    h = math.exp(-(x - c - d/2) / f)
    x = a + b / (1 + g) * (1 - 1 / (1 + h))
    return x / 104.981
    
def penalized_logp(mol):
    # Modified from https://github.com/bowenliu16/rl_graph_generation
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sa_scorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle
