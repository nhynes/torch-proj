Noise, parent = torch.class('nn.Noise', 'nn.Module')

Noise.updateOutput = (input) =>
  -- @output\resizeAs(input)\zero!
  @gradInput\randn(input\size!)
  @output\cmul(input, @gradInput)

Noise.updateGradInput = (input, gradOutput) => @gradInput\cmul(gradOutput)
