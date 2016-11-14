import dofile from require 'moonscript'
import thisfile from require 'paths'
require 'hdf5'
_ = require 'moses'
threads = require 'threads'
threads.serialization 'threads.sharedserialize'

formatters = dofile(thisfile 'formatters.moon')

export dataLoader

init = (opts) ->
  {:nworkers, :seed} = opts

  dsH5 = hdf5.open(opts.dataset, 'r')
  data = dsH5\all!
  dsH5\close!

  for partition in *{'train', 'val', 'test'}
    for fmt in *{formatters.groupByLen, formatters.maskUnk}
      fmt(data[partition], opts) if data[partition] ~= nil

  loaderOpts = _.omit(opts, 'savedState')
  loaderOpts.randState = (opts.savedState or {}).randState

  if nworkers > 0
    return threads.Threads nworkers,
      ->
        require 'moonscript'
        require 'torch'
        require 'loader.DataLoader',
      (tid) ->
        torch.setdefaulttensortype('torch.FloatTensor')
        if loaderOpts.randState
          torch.setRNGState(loaderOpts.randState[tid])
        else
          torch.manualSeed(seed+tid)
        dataLoader = DataLoader(data, loaderOpts)
  else
    require 'loader.DataLoader'
    dataLoader = DataLoader(data, loaderOpts)
    return {
      addjob: (f, cb) => cb f!
      synchronize: =>
      terminate: =>
    }

{ :init }
