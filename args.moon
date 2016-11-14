import dofile from require 'moonscript'
import thisfile from require 'paths'

DATA_DIR = thisfile 'data'

parse = (arg={}) ->
  cmd = with torch.CmdLine!
    \text!
    \text 'Options'

    \option '-seed', 4242, 'Manual random seed'
    \option '-gpu', 0, 'index of the GPU to use'

    -- Data
    \option '-dataset', DATA_DIR..'/dataset.h5', 'path to dataset'
    \option '-w2v', DATA_DIR..'/instructions_w2v.bin', 'path to w2v'
    \option '-nworkers', 2, 'number of data loading threads'
    \option '-batchSize', 75, 'mini-batch size'

    -- Model
    \option '-dim', 512, 'dimension'

    -- Training
    \option '-lr', 0.001, 'learning rate'
    \option '-optim', 'adam', 'optimizer to use (adam|sgd)'
    \option '-niters', -1, 'number of iterations for which to run (-1 is forever)'
    \option '-dispfreq', 100, 'number of iterations between printing train loss'
    \option '-valfreq', 500, 'number of iterations between validations'

    -- Saving & loading
    \option '-desc', '', 'description of what is being tested'
    \option '-savefreq', -1, 'number of iterations between snapshots (-1 is infinite)'
    \option '-saveafter', 0, 'how long before considering to save'
    \option '-loadsnap', '', 'file from which to load model'

    \text!

  cmd\parse arg

{ :parse }
