Din, parent = torch.class('nn.Din', 'nn.Container')
-- scales unit normal noise by standard deviations

-- Kingma, Diederik P., and Max Welling. “Auto-Encoding Variational Bayes.”
-- December 20, 2013. http://arxiv.org/abs/1312.6114.

Din.__init = (nInputPlanes) =>
  parent.__init(self)

  @sigma = with nn.Sequential! -- in: h (N x nInputPlanes x w x h)
    \add nn.Mean(3)
    \add nn.Mean(3)
    \add nn.Linear(nInputPlanes, nInputPlanes)
    \add nn.Exp!
    \add nn.HardTanh(0, 50)

  @l2 = nn.L2Penalty!
  @dkl = nn.DKLPenalty!

  @din = with nn.Sequential!  -- in: h, out: N(h, sigma)
    -- \add @l2
    \add with nn.ConcatTable!
      \add nn.Identity!
      \add with nn.Sequential!  -- out: N(0, sigma)
        \add @sigma
        \add @dkl
        \add nn.Noise!
        \add nn.View(-1, 1, 1)\setNumInputDims(1)
    \add nn.ExpandAs!
    \add nn.CAddTable!

  @modules = {@din}

  @fwds = 0

Din.updateOutput = (input) =>
  -- input: hidden params for each gaussian
  -- output: samples

  @fwds += 1 if @train
  if @fwds % 50 == 0
    print @sigma.output\mean!, @sigma.output\var!, input\mean!, input\var!

  @output = @din\forward(input)
  @output

Din.updateGradInput = (input, gradOutput) =>
  @gradInput = @din\backward(input, gradOutput)
  @gradInput
