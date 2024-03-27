#Input files

Can be found in 

`/eos/atlas/atlascerngroupdisk/phys-susy/pMSSM/Run2/MachineLearning/`

They are split in labelled/unlabelled and a second set merged_labelled where the labelled files are merged into bigger ones.

Each file contains 100 models, merged files 3000 models. Events are already shuffled in a file but only among the models considered in that chunk. This avoids the need to shuffle later, since it's not possible using the IterableDataset. There are about 4.2M labelled events and O(700M) unlabelled.

Events have already a very light skimming to remove collider-useless events: `MET + HT-jets + HT-photons + HT-bjets*2 + HT-leptons*10 > 300 GeV`
