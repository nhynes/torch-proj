_ = require 'moses'

import dofile from require 'moonscript'
import thisfile from require 'paths'

deepCopy = (tbl) ->
  copy = {k, (type(v) == 'table') and deepCopy(v) or v for k,v in pairs tbl}
  torch.setmetatable(copy, torch.typename(tbl)) if torch.typename(tbl)
  copy

git = dofile(thisfile '../gitdo.moon')

gitState = tostring(git\saveState!)\sub(1, 6)

(model, workers, opts, state) ->
  OUTFILE_TMP = 'snaps/'..opts.desc..'_'..gitState..'_i%s.t7'

  model = torch.isTypeOf(model, nn.DataParallelTable) and model\get(1) or model

  state.bestLoss = state.bestLoss or math.huge

  ->
    canSave = state.t >= opts.saveafter and state.valLoss < state.bestLoss
    if canSave
      state.bestLoss = state.valLoss

      outfile = string.format(OUTFILE_TMP, state.t)

      saveState = with _.pick(state, 't', 'optimState', 'bestLoss')
        .randState = [0]: torch.getRNGState!

      workers\specific(true)
      for i=1,workers.N
        workers\addjob i, (-> torch.getRNGState!), ((r) -> saveState.randState[i] = r)
      workers\specific(false)

      print 'Saving model to '..outfile..'...'
      snapModel = deepCopy(model)\float!\clearState!
      torch.save(outfile, {opts: opts, model: snapModel, state: saveState})

      collectgarbage!
