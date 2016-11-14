_ = require 'moses'

DataLoader = torch.class('DataLoader')

groupByLen = (data) ->
  slens = data.slens

  indsByLen = {}
  for i=1,slens\size(1)
    slen = slens[i]

    ibrl = indsByLen[slen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[slen] = ibrl

  lengths = _.keys indsByLen
  table.sort(lengths)

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = #indsByLen[lengths[i]] for i=1,#lengths -- freq of each index -> len

  with data
    .lengths = lengths
    .indsByLen = indsByLen
    .lenFreqs = lenFreqs

DataLoader.__init = (data, opts) =>
  @data = data
  @batchSize = opts.batchSize

DataLoader.makebatch = (partition='train', seed) =>
  randState = nil
  if seed ~= nil
    randState = torch.getRNGState!
    torch.manualSeed(seed)

  data = @data[partition] -- train, val, or test

  inp = torch.Tensor!
  tgt = torch.Tensor!

  collectgarbage!

  torch.setRNGState(randState) if randState ~= nil

  inp, tgt

DataLoader.partitionSize = (partition='train') => @data[partition].size

DataLoader.terminate = =>
