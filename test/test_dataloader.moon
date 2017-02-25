require 'dataloader'
require 'xlua'

dataLoader = DataLoader
  seed: 4242
  nworkers: 30
  batchSize: 100

for i,batch,n in dataLoader\run('train')
  {inp, tgt} = batch

  assert inp ~= nil and tgt ~= nil

  xlua.progress(i, dataLoader\size('train'))
