ExpandAs, parent = torch.class('nn.ExpandAs', 'nn.Module')
-- expands the second input to match the first

ExpandAs.__init = =>
  parent.__init(self)
  @output = {}
  @gradInput = {}

  @sum1 = torch.Tensor!
  @sum2 = torch.Tensor!

ExpandAs.updateOutput = (input) =>
  {a, b} = input

  @output[1] = a
  @output[2] = b\expandAs(a)

  @output

ExpandAs.updateGradInput = (input, gradOutput) =>
  b = input[2]
  {da, dbe} = gradOutput

  s1, s2 = @sum1, @sum2

  sumDst, sumSrc = s1, dbe
  for i=1,da\dim!
    if b\size(i) ~= da\size(i)
      sumDst\sum(sumSrc, i)
      sumSrc = sumSrc == s1 and s2 or s1
      sumDst = sumDst == s1 and s2 or s1

  @gradInput = {da, sumSrc}
  @gradInput
