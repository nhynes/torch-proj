L2Penalty, parent = torch.class('nn.L2Penalty', 'nn.Module')

L2Penalty.__init = =>
  parent.__init(self)
  @lambda = 1e-6

L2Penalty.updateOutput = (input) =>
  @output = input
  @output

L2Penalty.updateGradInput = (input, gradOutput) =>
  @gradInput\resizeAs(input)\copy(input)\mul(@lambda)\add(gradOutput)
