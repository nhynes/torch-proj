require 'cunn'
_ = require 'moses'

import dofile from require 'moonscript'
import thisfile from require 'paths'

DRIVERS = {'train', 'val', 'snap'}
driverInit = [dofile(thisfile driver..'.moon') for driver in *DRIVERS]

init = (model, workers, opts) ->
  state = _.defaults opts.savedState or {},
      t: 0
      crit: nn.MSECriterion!\cuda!

  state.optimState.learningRate = opts.lr if state.optimState ~= nil

  gpuTensors =
    gpuInp: torch.CudaTensor!

  state.prepBatch = (batchTensors) ->
    gt\resize(batchTensors[k]\size!)\copy(batchTensors[k]) for k, gt in pairs gpuTensors
    batchTensors = table.pack(...)
    for i=1,#gpuTensors
      gpuTensors[i]\resize(batchTensors[i]\size!)\copy(batchTensors[i])
    {imgs, noise}, {labelsReal, labelsFake, labelsReal}

  drivers, lazyDrivers = {}, {}
  for i, driver in pairs DRIVERS
    drivers[i] = (...) -> lazyDrivers[i](...)
    lazyDrivers[i] = (...) ->
      lazyDrivers[i] = driverInit[i](model, workers, opts, state)
      lazyDrivers[i](...)

  table.unpack drivers

{ :init }
