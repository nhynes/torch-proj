DISP_TMP = 'val iter %d |   loss: %-9.5g'

(model, workers, opts, state) ->
  {:prepBatch, :crit} = state

  valBatches = math.ceil(50000 / opts.batchSize)
  seeds = torch.LongTensor(valBatches)\random!

  ->
    model\evaluate!

    valLoss = 0

    for i=1,valBatches
      workers\addjob (-> dataLoader\makebatch 'val', seeds[i]),
        (...) ->
          input, target = prepBatch(...)
          model\forward(input)
          valLoss += crit\forward(model.output, target)

    workers\synchronize!

    valLoss /= valBatches
    state.valLoss = valLoss

    print string.format DISP_TMP, state.t, valLoss

    collectgarbage!
