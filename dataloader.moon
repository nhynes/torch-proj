require 'hdf5'
threads = with require 'threads'
  .serialization('threads.sharedserialize')

DataLoader = torch.class('DataLoader')

-------------------------------------------------------------------------------
-- edit these functions
loadData = =>
  dsH5 = hdf5.open('data/dataset.h5', 'r')
  data = dsH5\all!
  dsH5\close!

  data

getDatum = (data, i) ->
  input = torch.rand(3, 64, 64)
  target = torch.LongTensor{42}

  {input, target}

DataLoader._size = (part) => @data[part].\size(1)
-------------------------------------------------------------------------------

DataLoader.__init = (opt) =>
  @opt = opt

  data = loadData(self)
  @data = data
  collectgarbage!

  @workers = threads.Threads opt.nworkers, ->
    require 'torch'
    torch.setnumthreads(1)
    _G.data = data

batchLike = (data, batchSize) ->
  if type(data) == 'table'
    return [batchLike(datum, batchSize) for datum in *data]
  data.new(batchSize, table.unpack(data\size!\totable!))

copyIndex = (src, dest, idx) ->
  if type(src) == 'table'
    assert #src == #dest
    copyIndex(src[i], dest[i], idx) for i=1,#src
  else
    dest[idx] = src

makeBatch = (inds, part, ...) ->
  batchSize = inds\size(1)
  input, target = nil, nil
  for i=1,batchSize
    {inp, tgt} = getDatum(_G.data[part], inds[i])
    if input == nil
      input = batchLike(inp, batchSize)
      target = batchLike(tgt, batchSize)
    copyIndex(inp, input, i)
    copyIndex(tgt, target, i)
  collectgarbage!
  {input, target}, ...

DataLoader.size = (part='train') => math.ceil(@_size(part) / @opt.batchSize)

DataLoader.run = (part='train', maxBatches=math.huge) =>
  perm =  torch.LongTensor!
  part == 'train' and perm\randperm(@_size(part)) or perm\range(1, @_size(part))
  batchInds = perm\split(@opt.batchSize)
  maxBatches = math.min(maxBatches, #batchInds)

  n, batch, batchno = 1, nil, nil
  enqueue = ->
    while n <= maxBatches and @workers\acceptsjob!
      @workers\addjob makeBatch,
        ((_batch, _n) ->
          batch = _batch
          batchno = _n),
        batchInds[n], part, n
      n += 1

  loader = (i) =>
    i += 1
    enqueue!
    if not @workers\hasjob!
      return nil
    @workers\dojob!
    if @workers\haserror!
      @workers\synchronize!
    enqueue!
    i, batch, batchno

  loader, self, 0
