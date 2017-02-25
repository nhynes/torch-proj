require 'model.init'

model = Model
  net: 'model'
inp = torch.rand(2, 1, 110000, 1)
model\forward(inp)
