VarEmb, parent = torch.class('nn.VarEmb', 'nn.Container')

STEP = 5

VarEmb.__init = (inputDim) =>
  parent.__init(self)

  -- @range = torch.range(0, -inputDim/10+0.1, -0.1)\view(1, -1)
  @range = torch.range(0, -inputDim*STEP+STEP, -STEP)\view(1, -1)

  @x0 = with nn.Sequential!   -- input: x, output: x0
    \add nn.Linear(inputDim, inputDim)
    \add cudnn.Tanh!
    \add nn.Linear(inputDim, inputDim/2)
    \add cudnn.Tanh!
    \add nn.Linear(inputDim/2, 1)
    -- \add nn.Abs!
    \add nn.Exp!

  @mask = with nn.Sequential! -- input: {range, x0}, output: 1/(1 + exp(x0-x))
    \add with nn.ParallelTable!
      \add nn.Identity!
      \add nn.Replicate(inputDim, 1, 1)
    \add nn.CAddTable!
    \add cudnn.Sigmoid!

  @varemb = with nn.Sequential!
    \add nn.CMulTable! -- input: {x, mask}

  @forwards = 0

  @modules = {@x0, @mask, @varemb}

VarEmb.updateOutput = (input) =>
  if @forwards == 0
    @x0\get(5).bias\add(math.log(input\size(2)*STEP))

  @x0\forward(input)
  @mask\forward{@range\expandAs(input), @x0.output}
  @output = @varemb\forward{input, @mask.output}

  if @forwards % 50 == 0 and @train
    print @x0.output\max!/STEP, @x0.output\min!/STEP

  @forwards += 1 if @train

  @output

VarEmb.updateGradInput = (input, gradOutput) =>
  {dX, dMask} = @varemb\backward({input, @mask.output}, gradOutput)
  {dRange, dX0} = @mask\backward({@range\expandAs(input), @x0.output}, dMask)
  reg = 0
  if @forwards > 500
    reg = 1e-7
  -- if @forwards > 1000
  --   reg = 1e-6
  -- if @forwards > 1500
  --   reg = 1e-5
  -- if @forwards > 3000
  --   reg = 2e-5
  -- if @forwards > 5000
  --   reg = 3e-5
  -- if @forwards > 1500
  --   reg = 1e-1
  dX0\add(reg)
  @x0\backward(input, dX0)
  @gradInput = dX\add(@x0.gradInput)
  @gradInput
