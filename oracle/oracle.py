from numpy import isin
import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader

from .scorer.scorer import get_scores

class Oracle():
    def __init__(self, args, mols_ref=None):
        '''
        @params:
            args (dict): argsurations
        '''
        self.objectives = args.objectives
        self.fps_ref = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) 
                        for x in mols_ref] if mols_ref else None
        self.device = torch.device(args.device)

    def batch_get_scores(self, mols):
        '''
        @params:
            mols: molecules to estimate score
        @return:
            dicts (list): list of score dictionaries
        '''
        dicts = [{} for _ in mols]
        for obj in self.objectives:
            scores = get_scores(obj, mols, device=self.device)
            for i, mol in enumerate(mols):
                dicts[i][obj] = scores[i]
        return dicts
    
    def get_score(self, mol):
        scores = {}
        for obj in self.objectives:
            score = get_scores(obj, mol, device=self.device)
            scores[obj] = score[0]
    
        return scores