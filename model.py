import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import model_blocks as mb

'''
SimCLR loss for contrastive learning https://arxiv.org/abs/2002.05709
code copied from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
'''
def SimCLR(feats, temperature=1):

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool) #, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    #print(self_mask, cos_sim.shape[0]//2)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0) # this needs to be updated
    #print(pos_mask)
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    return nll

class Model(pl.LightningModule):

    def __init__(
        self,
        embed_dimensions,
        embed_normalize_input,
        bkg_dimensions,
        bkg_normalize_input,
        lr = 1e-3,
        weights = None,
	):
        
        super().__init__()

        # model
        self.model = mb.SignatureDiscovery(embed_dimensions, embed_normalize_input, bkg_dimensions, bkg_normalize_input)
        
        # other
        self.lr = lr

        # use the weights hyperparameters
        if weights: 
            ckpt = torch.load(weights,map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])
        
        self.save_hyperparameters()

    def forward(self, x):

        emb, bkg = self.model(x)
        return emb, bkg
        
    def step(self, batch, batch_idx, version, dataloader_idx=0):
        
        # run model
        x, y = batch

        # compute loss
        loss = self.loss(embedding, bkg, y)

        # log the loss
        if dataloader_idx==0:
            for key, val in loss.items():
                self.log(f"{version}_{key}", val, prog_bar=(key=="loss"), on_step=(version=="train"))
        
        return loss["loss"]
    
    def training_step(self, batch, batch_idx, debug=False):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx, debug=False):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def loss(self, emb, bkg, y):

        # total loss
        l = {}

        # cdistance
        l["contrastive"] = SimCLR(feats=emb, temperature=1)

        # background estimate
        l["bkg"] = torch.nn.MSELoss(bkg, y)

        # get total
        l['loss'] = sum(l.values())

        return l

if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    # load file and events
    df = pd.read_hdf("data/testfile_files100_35.h5")
    n = 100
    # inputs
    x = [i for i in df.columns if "201" not in i and "isinSR" not in i and "model" not in i]
    x = torch.Tensor(np.array(df[:n][x]))
    y = [i for i in df.columns if "201" in i]
    y = torch.Tensor(np.array(df[:n][y]))
    print(x.shape, y.shape)
    # x = torch.Tensor(3,10) # batch x features
    # y = torch.Tensor([ # test 4 SR's
    #     [1, 0, 1, 0],
    #     [0, 0, 1, 1],
    #     [0, 1, 1, 1]
    # ])
    # print(y.shape)
    m = Model([x.shape[1], 10, 5], False, [5, 3, 1], False)
    emb, bkg = m(x)
    print(emb.shape, bkg.shape)

    # SimCLR
    # temperature
    temperature = 1
    nll = SimCLR(feats=emb, temperature=1)
    print(nll)

