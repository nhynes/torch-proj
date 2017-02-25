require 'dpnn'

Model, parent = torch.class('Model', 'nn.Container')

Model.__init = (opt) =>
  parent.__init(self)

  if opt.nGPU
    require 'cunn'
    require 'cudnn'

  init = nil
  if opt.model == 'model'
    init = require('model.model')

  return init(self, opt)

Model.updateOutput = (input) =>
  @output = @model\forward(input)
  @output

Model.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput

------------------------------------------------------------

nn.Module.dontTrain = =>
  @parameters = =>
  @accGradParameters = =>
  @dpnn_getParameters_found = true
  self

nn.Container.dontTrain = =>
  @applyToModules (mod) -> mod\dontTrain!
  nn.Module.dontTrain(self)
