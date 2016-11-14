Model, parent = torch.class('Model', 'nn.Container')

Model.__init = (opts) =>
  parent.__init(self)

  @model = nn.Identity!

  @modules = {@model}

Model.updateOutput = (input) =>
  [==[
  input:
  output:
  ]==]
  @output = @model\forward(input)
  @output

Model.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
