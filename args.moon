(arg={}) ->
  cmd = with torch.CmdLine!
    \text!
    \text 'Options'

    \option '-seed',      4242,   'Manual random seed'

    -- Data
    \option '-nworkers',  4,      'number of data loading threads'
    \option '-batchSize', 64,     'mini-batch size'

    -- Model
    \option '-net',       'model','model family (sound|wave|deconv)'
    \option '-filtx',      1,     'filter size multiplier'

    -- Training
    \option '-optim',     'adam', 'optimizer to use (adam|sgd)'
    \option '-lr',        0.001,  'learning rate'
    \option '-epochs',    -1,     'number of epochs for which to run (-1 is forever)'
    \option '-dispfreq',  100,    'number of iterations between printing train loss'
    \option '-nval',      1000,   'number of validation batches'

    -- Saving & loading
    \option '-desc',      '',     'description of experiment'
    \option '-savefreq',  -1,     'how often to checkpoint (-1 is never)'
    \option '-saveafter', 0,      'epoch after which to start checkpointing'
    \option '-loadsnap',  '',     'load state from checkpoint'

    \text!

  opt = cmd\parse arg

  assert opt.desc ~= '', 'desc must not be empty'

  arg
