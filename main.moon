require 'torch'
require 'cutorch'
require 'moonscript'
posix = require 'posix'

require 'model.init'
require 'dataloader'
require 'drivers'

opt = require('args')(arg)

opt.nGPU = cutorch.getDeviceCount!

dataLoader = DataLoader(opt)

model = nil
if paths.filep opt.loadsnap
  print 'Loading model from '..opt.loadsnap
  _ = require 'moses'
  snap = torch.load(opt.loadsnap)
  model = snap.model

  newopt = _.pick opt, 'loadsnap', 'epochs', 'dispfreq', 'savefreq', 'lr'
  opt = _.extend(snap.opt, newopt)
  opt.savedState = snap.state

  torch.setRNGState(opt.savedState.randState[0])

  print 'Resuming training from iteration '..opt.savedState.t
else
  torch.manualSeed(opt.seed)
  model = Model(opt)

if opt.nGPU > 1
  model = with nn.DataParallelTable(1, true, true)
    \add model
    \threads -> require 'model.init'

model\cuda!

crit = with nn.ParallelCriterion!
  \add nn.DistKLDivCriterion!
  \add nn.DistKLDivCriterion!
  \cuda!

drivers = Drivers(model, crit, dataLoader, opt)

paths.mkdir('run')
RUNFILE = 'run/'..opt.desc
posix.mkfifo(RUNFILE)
runfd = posix.open(RUNFILE, bit.bor(posix.O_RDONLY, posix.O_NONBLOCK))

drivers\addHook 'iter', (drivers) ->
  cmd = stringx.strip(posix.read(runfd, 100))
  if cmd == 'snap'
    drivers\snap(0)

collectgarbage!

drivers\run(t) for t=1,(opt.epochs > 0 and opt.epochs or math.huge)
