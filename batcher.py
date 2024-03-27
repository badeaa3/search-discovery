'''
Author: Anthony Badea
Date: March 18, 2023
'''

import pandas as pd
import torch

class pMSSMDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_files):
        super(pMSSMDataset).__init__()
        self.data_files = data_files
        self.SRs = None
        self.features = None

    def __iter__(self):
        for data_file in self.data_files:
            df = pd.read_hdf(data_file)
            if not self.SRs:
                self.SRs = [x for x in df.columns if x.startswith("SR")]
                self.features = [x for x in df.columns if x.startswith("feature")]
                
            for x,y in zip(df[self.features].itertuples(index=False), df[self.SRs].itertuples(index=False)):
                yield torch.Tensor(x),torch.Tensor(y).type(torch.uint8)

    def worker_init_fn(worker_id):
        '''
        The whole dataset is given to each worker, need this function to avoid duplication of data across workers
        Make sure to use as argument when building the DataLoader
        '''
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        # keep only a subset of data_files
        dataset.data_files = [f for i,f in enumerate(dataset.data_files) if i%worker_info.num_workers == worker_id]

def train_test_datasets(infiles, test_size):
    '''
    Provide an approximate splitting based on number of files
    '''
    test_modulo = int(1/test_size)
    test_files  = [f for i,f in enumerate(infiles) if i%test_modulo==0]
    train_files = [f for i,f in enumerate(infiles) if i%test_modulo!=0]
    return pMSSMDataset(train_files),  pMSSMDataset(test_files)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFiles", help="Input training file.", nargs="+")
    ops = parser.parse_args()

    from torch.utils.data import DataLoader

    ds = pMSSMDataset(ops.inFiles)
    iterloader = iter(DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=pMSSMDataset.worker_init_fn))
    for _ in range(5):
        print(next(iterloader))
