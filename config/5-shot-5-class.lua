
return function(opt)
   opt.nClasses = {train=10, val=10, test=10}
   opt.nTrainShot = 5
   opt.nEval = 15
   
   opt.nTest = {100, 250, 600}
   opt.nTestShot = {5}

--   opt.paramsFile = 'metaLearner_params_snapshot.th'

   return opt
end
