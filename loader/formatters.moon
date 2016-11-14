_ = require 'moses'

groupByLen = (data) ->
  sentLens = data.sentLens

  indsByLen = {}
  for i=1,sentLens\size(1)
    ilen = sentLens[i]

    ibrl = indsByLen[ilen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[ilen] = ibrl

  lengths = _.keys indsByLen
  for len, inds in pairs indsByLen
    indsByLen[len] = torch.LongTensor(inds)
  table.sort lengths

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = indsByLen[lengths[i]]\size(1) for i=1,#lengths

  data.lengths = lengths
  data.indsByLen = indsByLen
  data.lenFreqs = lenFreqs

UNK = 1
maskUnk = (data, opts) ->
  data.sents\maskedFill(data.instrs\gt(opts.vocabSize), UNK)

{:groupByLen, :maskUnk}
