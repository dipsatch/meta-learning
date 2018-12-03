local _ = require 'moses'

function check(rawData, opt, k)
   if #rawData.item <= 0 or (opt.nClasses[k] 
      and #rawData.item < opt.nClasses[k]) then
      
      return false
   end  

   return true
end

return function(opt)

   local dataOpt = {
      cuda = opt.useCUDA,
      episodeSamplerKind = opt.episodeSamplerKind or 'permutation',  

      dataCacheDir = opt.rawDataDir,

      nClass = opt.nClasses,
      nSupportExamples = math.max(opt.nTrainShot, math.max(unpack(opt.nTestShot))),
      nEvalExamples = opt.nEval,

      imageDepth = opt.nDepth,
      imageHeight = opt.nIn,
      imageWidth = opt.nIn,
      resizeType = 'scale',
      
      normalizeData = opt.normalizeData
   }
   
   local data = require(opt.dataName)(dataOpt)
   package.path = package.path .. ';/home/ec2-user/uic/meta-learning-lstm/dataset/dataset2.lua'

   require('Dataset2')
   
   local function prepareDataset(split, sample, field, batchSize)
      local examples = torch.cat(_.map(sample.item, function(item, i)
         return item.extra[field]
      end), 1) 

      local classes
      if opt.trainFull and split == 'train' then
         classes = torch.cat(_.map(sample.item, function(item, i)
            return item.extra[field].new(item.extra[field]:size(1)):fill(item.url)
         end), 1)
      else
         classes = torch.cat(_.map(sample.item, function(item, i)
            return item.extra[field].new(item.extra[field]:size(1)):fill(i)
         end), 1)
      end
      local ds = Dataset2({ x = examples, y = classes, batchSize=batchSize, shuffle=true})

      return ds 
   end

   _.each(data, function(v, k)
      if type(data[k]) == 'table' and data[k].get then
         data[k].createEpisode = 
            function(lopt)
               local rawData = data[k].get()
               while not check(rawData, opt, k) do
                  rawdata = data[k].get()
               end
               local trainDataset, testDataset = prepareDataset(k, 
                  rawData, 'supportExamples', lopt.trainBatchSize), 
                  prepareDataset(k, rawData, 'evalExamples', lopt.testBatchSize)
                              
               return trainDataset, testDataset
            end 
      end 
   end)  

   return data.train, data.val, data.test
end
