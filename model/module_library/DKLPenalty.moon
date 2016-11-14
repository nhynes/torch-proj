DKLPenalty, parent = torch.class('nn.DKLPenalty', 'nn.Module')

DKLPenalty.__init = =>
  parent.__init(self)
  @lambda = 1e-7

DKLPenalty.updateOutput = (input) =>
  @output = input
  @output

DKLPenalty.updateGradInput = (input, gradOutput) =>
  -- loss = (x^2 - log x^2)/2
  -- @gradInput\resizeAs(input)\copy(input)\add(1e-8)\cinv!\mul(-1)\add(input)
  @gradInput\resizeAs(input)\copy(input)\add(1e-8)\cinv!\mul(-1)
  @gradInput\mul(@lambda)\add(gradOutput)
