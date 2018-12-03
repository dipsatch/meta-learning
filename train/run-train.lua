#!/usr/bin/env torch

require 'pl'

-- options
local opt = lapp[[
   Arguments for training k-shot meta-learner 
   
   --task               (default -)                path to config file for task 
   --data               (default -)                path to config file for data
   --model              (default -)                path to config file for model
   --test               (default -)                path to config file for test details
]]
--[[
local opt = {}
opt['task'] = 'config.1-shot-5-class'
opt['data'] = 'config.imagenet'
opt['model'] = 'config.lstm.train-imagenet-1shot'
opt['test'] = '-'
--]]

-- run
local result = require('train.train')(opt)
print('Task: ' .. opt.task .. '; Data: ' .. opt.data .. '; Model: ' .. opt.learner)
print('Test Accuracy: ')
print(result)

