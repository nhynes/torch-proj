require 'nn'

mwpPass = (input, exOutput) => @mwp\resizeAs(input)\copy(exOutput)

initAffine = =>
   @exWeight = torch.Tensor!
   @exOutput = torch.Tensor!

mwpAffine = (input, exOutput, contrasting) =>
   if @contrastive and contrasting
      @exWeight\mul(@weight, -1)\clamp(0, math.huge)
   else
      @exWeight\clamp(@weight, 0, math.huge)

   _weight = @weight
   @weight = @exWeight -- set this before calling

   _bias = @bias
   @bias = nil

   _output = @output
   @output = @exOutput
   @output\resizeAs(_output)

   self\updateOutput(input)

   @exOutput\add(1e-10)\cinv!\cmul(exOutput)

   nn.Module.exBackward(self, input, @exOutput)

   @bias = _bias
   @weight = _weight
   @output = _output

   @mwp\cmul(input)


setup = (moduleName, fns) ->
   ctor = torch.getconstructortable(moduleName)
   for fn, impl in pairs fns
      if ctor[fn] ~= nil
         _prevFn = ctor[fn]
         ctor[fn] = (...) =>
            _prevFn(self, ...)
            impl(self, ...)
      else
         ctor[fn] = impl


setup 'nn.Module',
   __init: => @mwp = torch.Tensor!
   exBackward: (input, exOutput) =>
      _gradInput = @gradInput
      @gradInput = @mwp
      self\updateGradInput(input, exOutput)
      @gradInput = _gradInput
      @mwp
   setContrastive: (contrastive=true) =>
      @contrastive = contrastive
      self


setup 'nn.Sequential',
   exBackward: (input, exOutput, contrasting) =>
      currentExcitationOutput = exOutput
      currentModule = @modules[#@modules]

      for i=#@modules-1,1,-1
         previousModule = @modules[i]
         currentExcitationOutput = self\rethrowErrors currentModule, i+1, 'exBackward',
            previousModule.output, currentExcitationOutput, contrasting
         currentModule = previousModule

      @mwp = self\rethrowErrors currentModule, 1, 'exBackward',
         input, currentExcitationOutput, contrasting

      @mwp


setup 'nn.SpatialConvolution', {__init: initAffine, exBackward: mwpAffine}
setup 'nn.Linear', {__init: initAffine, exBackward: mwpAffine}

if cudnn
   setup 'cudnn._Pointwise', exBackward: mwpPass
   setup 'cudnn.BatchNormalization', exBackward: mwpPass

setup 'nn.BatchNormalization', exBackward: mwpPass
setup 'nn.Sigmoid', exBackward: mwpPass
