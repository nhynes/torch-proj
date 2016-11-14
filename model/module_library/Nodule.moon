Nodule, parent = torch.class('nn.Nodule', 'nn.Module') -- aNonymous Module

Nodule.__init = (updateOutput, updateGradInput) =>
  parent.__init(self)
  @updateOutput = updateOutput
  @updateGradInput = updateGradInput

Nodule.updateOutput = (input) =>
  if @updateOutput ~= nil
    @updateOutput(self, input)
  else
    @output = input
  @output

Nodule.updateGradInput = (input, gradOutput) =>
  if @updateGradInput ~= nil
    @updateGradInput(self, input, gradOutput)
  else
    @gradInput = gradOutput
  @gradInput
