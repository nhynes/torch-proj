require 'torch'
require 'os'

import dofile from require 'moonscript'
import thisfile from require 'paths'

math.randomseed(os.time!)

dofile(thisfile 'DataLoader.moon')

opts =
  batchSize: 2

data =
  train: torch.Tensor!
  val: torch.Tensor!

dl = DataLoader(data, opts)

print(dl\makebatch!)
