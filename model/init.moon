require 'cunn'
require 'cudnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'LookupTableW2V.moon')
dofile(thisfile 'Model.moon')

init = (opts) -> Model

nn.Module.dontTrain = =>
  @parameters = =>
  @accGradParameters = =>
  @dpnn_getParameters_found = true
  self

nn.Container.dontTrain = =>
  @applyToModules (mod) -> mod\dontTrain!
  nn.Module.dontTrain(self)

{ :init }
