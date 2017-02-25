require 'optim'
require 'sys'

Drivers, parent = torch.class('Drivers')

s2t = f: '', h: 'Half', d: 'Double', l: 'Long', i: 'Int'
makeTensors = (spec, pin=false) ->
  if type(spec) == 'table'
    return [makeTensors(s, pin) for s in *spec]
  torch['Cuda'..(pin and 'Host' or '')..s2t[spec]..'Tensor']!

copyInputs = (src, dest) ->
  if type(src) == 'table'
    assert #src == #dest
    for i=1,#src
      copyInputs(src[i], dest[i])
  else
    dest\resize(src\size!)\copy(src)

deepCopy = (tbl) ->
  copy = {k, (type(v) == 'table') and deepCopy(v) or v for k,v in pairs tbl}
  torch.setmetatable(copy, torch.typename(tbl)) if torch.typename(tbl)
  copy

Drivers.__init = (model, crit, dataLoader, opt) =>
  @model, @crit, @dataLoader, @opt = model, crit, dataLoader, opt

  if opt.savedState
    @optimState = with opt.savedState.optimState
      .learningRate = opt.lr
  else
    @optimState = learningRate: opt.lr
    if opt.optim == 'sgd'
      with @optimState
        .momentum = 0.9
        .nesterov = true
        .dampening = 0
    else
      with @optimState
        .beta = 0.9
        .beta2 = 0.999
        .epsilon = 1e-8
  @params, @gradParams = model\getParameters!
  f = -> @crit.output, @gradParams
  @optimize = => optim[opt.optim](f, @params, @optimState)

  @gpuTensors = makeTensors({'f', {'f', 'f'}}, opt.nGPU > 1)

  @hooks = {}

Drivers.train = (t) =>
  @model\training!
  hook(self, t) for hook in *(@hooks.train or {})
  loss, timer = 0, 0
  for i,batch in @dataLoader\run('train')
    sys.tic!
    hook(self, t, i, batch) for hook in *(@hooks.iter or {})
    copyInputs(batch, @gpuTensors)
    {input, target} = @gpuTensors
    @model\forward(input)
    loss += @crit\forward(@model.output, target)
    @crit\backward(@model.output, target)
    @model\zeroGradParameters!
    @model\backward(input, @crit.gradInput)
    @optimize!
    timer += sys.toc!

    assert @params\storage! == @model\parameters![1]\storage!

    if @opt.dispfreq > 0 and i % @opt.dispfreq == 0
      print string.format '%s[%d] (%d/%d) | loss: %g\t(%.2f)',
        i / @opt.dispfreq == 1 and '\n' or '',
        t, i, @dataLoader\size('train'),
        loss / @opt.dispfreq, timer / @opt.dispfreq
      loss = 0
      timer = 0

Drivers.val = (t) =>
  @model\evaluate!
  hook(self, t) for hook in *(@hooks.val or {})
  n,loss = 0
  for i,batch in @dataLoader\run('val', @opt.nval)
    hook(self, t, i, batch) for hook in *(@hooks.valIter or {})
    copyInputs(batch, @gpuTensors)
    {input, target} = @gpuTensors
    @model\forward(input)
    loss += @crit\forward(@model.output, target)
    n += 1

  loss /= n
  @valLoss = loss
  print string.format '[%d] (VAL) | loss: %g', t, loss

Drivers.snap = (t) =>
  @model\training!
  hook(self, t) for hook in *(@hooks.snap or {})

  model = torch.isTypeOf(@model, nn.DataParallelTable) and @model\get(1) or @model
  snapModel = deepCopy(model)\float!\clearState!

  snapdir = 'snaps/'..@opt.desc
  paths.mkdir(snapdir)
  torch.save string.format('%s/model_%s.t7', snapdir, t), {opt: @opt, model: snapModel}
  torch.save string.format('%s/state_%s.t7', t),
    {optimState: @optimState, randState: torch.getRNGState!}

  collectgarbage!

Drivers.run = (t) =>
  @train(t)
  @val(t)
  @snap(t)

Drivers.addHook = (hook, handler) =>
  hooks = @hooks[hook] or {}
  hooks[#hooks+1] = handler
  @hooks[hook] = hooks
