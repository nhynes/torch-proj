require 'torch'
require 'cutorch'
require 'drivers.CallbackQueue'

import dofile from require 'moonscript'
import thisfile from require 'paths'

git = dofile(thisfile 'gitdo.moon')

args = require 'args'
opts = args.parse arg

if paths.filep opts.loadsnap
  snapDesc = opts.loadsnap\split('_')
  gitState = snapDesc[#snapDesc-1]
  git.pushState(gitState)

model = require 'model.init'
loader = require 'loader.init'
drivers = require 'drivers.init'

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opts.gpu+1)

Model = model.init(opts)
theModel = nil
if paths.filep opts.loadsnap
  print 'Loading model from '..opts.loadsnap
  _ = require 'moses'
  snap = torch.load(opts.loadsnap)
  theModel = snap.model

  newOpts = _.pick opts,
    'loadsnap', 'niters', 'dispfreq', 'valfreq', 'savefreq', 'lrG', 'lrD'
  opts = _.extend(snap.opts, newOpts)
  opts.savedState = snap.state

  torch.setRNGState(opts.savedState.randState[0])

  print 'Resuming training from iteration '..opts.savedState.t
else
  torch.manualSeed(opts.seed)
  theModel = Model(opts)

cudnn.convert(theModel\cuda!, cudnn)
theModel\apply (mod) ->
  if torch.type(mod)\find('Convolution')
    mod\setMode 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1'

workers = loader.init(opts)

train, val, snap = drivers.init(theModel, workers, opts)

done = ->
  workers\addjob (-> dataLoader\terminate!), ->
  workers\terminate!
  os.exit!

-- set up callbacks
cbq = with CallbackQueue(opts.startiter)
  \add cb: done, iter: opts.niters > 0 and opts.niters or math.huge, priority: -math.huge
  \add cb: val, interval: opts.valfreq, iter: opts.valfreq, priority: math.huge if opts.valfreq > 0
  \add cb: snap, interval: opts.savefreq, iter: opts.savefreq if opts.savefreq > 0

if paths.filep opts.loadsnap
  git.popState!

collectgarbage!

-- val!
while #cbq > 0
  train! for t=1,cbq\waitTime!
  workers\synchronize!
  cbq\advance!
  cb! for cb in cbq\pull!
