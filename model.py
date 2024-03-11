import torch
import pytorch_lightning as pl
import model_blocks as mb

class Model(pl.LightningModule):

    def __init__(
        self,
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

        embedding, bkg = self.model(x)
        return embedding, bkg
        
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

    def loss(self, loss, mu_logvar, xloss, randloss, candidates_p4, jet_choice):

        # total loss
        l = {}
        l["distances"]   =  0
        
        # get total
        l['loss'] = sum(l.values())

        return l
