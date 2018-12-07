local t = require 'torch'
local autograd = require 'autograd'
local nn = require 'nn'
local util = require 'util.util'
local _ = require 'moses'

require 'model.lstm.meta-learner-lstm'

return function(opt, dataset)
   -- data
   local metaTrainSet = dataset.train
   local metaValidationSet = dataset.validation
   local metaTestSet = dataset.test
 
   -- keep track of errors
   local avgs = {} 
   local trainConf = optim.ConfusionMatrix(opt.nClasses.train)
   local valConf = {}
   local testConf = {}
   for _,k in pairs(opt.nTestShot) do
      valConf[k] = optim.ConfusionMatrix(opt.nClasses.val)
      testConf[k] = optim.ConfusionMatrix(opt.nClasses.test)
      avgs[k] = 0 
   end 

   -- learner
   local learner = getLearner(opt)  
   print("Learner nParams: " .. learner.nParams)   

   -- meta-learner     
   local metaLearner = getMetaLearner({learnerParams=learner.params, 
      nParams=learner.nParams, debug=opt.debug, 
      homePath=opt.homePath, nHidden=opt.nHidden, BN1=opt.BN1, BN2=opt.BN2})  
   local classify = metaLearner.f 
     
   -- load params from file?
   if opt.paramsFile then
      print("loading params from: " .. opt.paramsFile)
      local loadedParams = torch.load(opt.paramsFile)
      metaLearner.params = loadedParams
   end

   -- cast params to float or cuda 
   local cast = "float"
   if opt.useCUDA then
      cast = "cuda"
   end
   metaLearner.params = autograd.util.cast(metaLearner.params, cast)
   print("Meta-learner params")
   print(metaLearner.params)        

   local nEpisode = opt.nEpisode
   local cost = 0
   local timer = torch.Timer()
   local printPer = opt.printPer 
   local evalCounter = 1
   local prevIterParams

   local lstmState = {{},{}}

   ---------------------------------------------------------------------------- 
   -- meta-training

   -- init optimizer
   local optimizer, optimState = autograd.optim[opt.optimMethod](
      metaLearner.dfWithGradNorm, tablex.deepcopy(opt), metaLearner.params) 

   -- episode loop 
   for d=1,nEpisode do  
      -- create training epsiode 
      local trainSet, testSet = metaTrainSet.createEpisode({})
   
      -- train on meta-train 
      local trainData = trainSet:get()
      local trainInput, trainTarget = util.extractK(trainData.input, 
         trainData.target, opt.nTrainShot, opt.nClasses.train)
      local testData = testSet:get()
  
      local gParams, loss, prediction = optimizer(learner, trainInput, 
         trainTarget, testData.input, testData.target, 
         opt.nEpochs[opt.nTrainShot], opt.batchSize[opt.nTrainShot])
      cost = cost + loss      
      
      for i=1,prediction:size(1) do
         trainConf:add(prediction[i], testData.target[i])   
      end 

      -- status check of meta-training & evaluate meta-validation 
      if math.fmod(d, printPer) == 0 then
         local elapsed = timer:time().real
         print(string.format(
            "Dataset: %d, Train Loss: %.3f, LR: %.3f, Time: %.4f s", 
            d, cost/(printPer), util.getCurrentLR(optimState[1]), elapsed))
         print(trainConf) 
         trainConf:zero()

         local accVal = {}
         for _, k in pairs(opt.nTestShot) do
            accVal[k] = torch.zeros(opt.nValidationEpisode)
         end

         -- meta-validation loop   
         for v=1,opt.nValidationEpisode do
            local trainSet, testSet = metaValidationSet.createEpisode({})
            local trainData = trainSet:get()
            local testData = testSet:get()
            
            -- k-shot loop
            for _,k in pairs(opt.nTestShot) do
               local trainInput, trainTarget = util.extractK(trainData.input, 
                  trainData.target, k, opt.nClasses.val)
   
               local _, prediction = classify(metaLearner.params, learner, 
                  trainInput, trainTarget, testData.input, testData.target, 
                  opt.nEpochs[k] or opt.nEpochs[opt.nTrainShot], 
                  opt.batchSize[k] or opt.batchSize[opt.nTrainShot], true)     
      
               for i=1,prediction:size(1) do
                  valConf[k]:add(prediction[i], testData.target[i])  
               end
               valConf[k]:updateValids()
               accVal[k][v] = valConf[k].totalValid*100
            end
         end   
         
         -- print accuracy on meta-validation set 
         for _,k in pairs(opt.nTestShot) do 
            print('Validation Accuracy (' .. opt.nValidationEpisode 
               .. ' episodes, ' .. k .. '-shot)')
            print(accVal[k]:mean())
            print(valConf[k])
            valConf[k]:zero()
         end
   
         cost = 0
         timer = torch.Timer() 
      end

      if math.fmod(d, 1000) == 0 then
         local prevIterParams = util.deepClone(metaLearner.params)   
         torch.save("metaLearner_params_snapshot.th", 
            autograd.util.cast(prevIterParams, "float"))
      end   
   end
--]]
   ----------------------------------------------------------------------------
   -- meta-testing
   local ret = {} 
   -- number of episodes loop
   _.each(opt.nTest, function(n, i)
      local acc = {}
      for _, k in pairs(opt.nTestShot) do
         acc[k] = torch.zeros(n)
      end
      
      -- episodes loop 
      for d=1,n do 
         local trainSet, testSet = metaTestSet.createEpisode({})   

         local trainData = trainSet:get() 
         local testData = testSet:get()
         local loss, prediction 

         -- k-shot loop
         for _, k in pairs(opt.nTestShot) do 
            local trainInput, trainTarget = util.extractK(trainData.input, 
               trainData.target, k, opt.nClasses.test)

--[[
              local gParams, loss, prediction = optimizer(learner, trainInput,
                 trainTarget, testData.input, testData.target,
                 opt.nEpochs[k], opt.batchSize[k])
--]]       
            local loss, prediction = classify(metaLearner.params, learner, 
               trainInput, trainTarget, testData.input, testData.target, 
               opt.nEpochs[k] or opt.nEpochs[opt.nTrainShot], 
               opt.batchSize[k] or opt.batchSize[opt.nTrainShot], true)
            
            for i=1,prediction:size(1) do
               testConf[k]:add(prediction[i], testData.target[i])
            end

            testConf[k]:updateValids()  
            acc[k][d] = testConf[k].totalValid*100
--            testConf[k]:zero()
         end
      end

      for _,k in pairs(opt.nTestShot) do 
         print('Test Accuracy (' .. n .. ' episodes, ' .. k .. '-shot)')
         print(acc[k]:mean())
         print(testConf[k])
         testConf[k]:zero()
      end
--      os.exit()
 
      ret[n] = _.values(_.map(acc, function(val, i) 
            local low = val:mean() - 1.96*(val:std()/math.sqrt(val:size(1)))
            local high = val:mean() + 1.96*(val:std()/math.sqrt(val:size(1)))       
            return i .. '-shot: ' .. val:mean() .. '; ' .. val:std() 
               .. '; [' .. low .. ',' .. high .. ']' 
      end))
   end)

   return ret
end
