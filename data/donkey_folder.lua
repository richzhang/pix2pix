
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
print(os.getenv('DATA_ROOT'))
opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)

if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local output_nc = opt.output_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local preprocessAandB = function(imA, imB)
  imA = image.scale(imA, loadSize[2], loadSize[2])
  imB = image.scale(imB, loadSize[2], loadSize[2])
  local perm = torch.LongTensor{3, 2, 1}
  imA = imA:index(1, perm)--:mul(256.0): brg, rgb
  imA = imA:mul(2):add(-1)
  imB = imB:index(1, perm)
  imB = imB:mul(2):add(-1)
--   print(img:size())
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")
 
  
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imA:size(2)
  local iW = imA:size(3)
  
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end
  
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end
  
  return imA, imB
end



local function loadImageChannel(path)
    local input = image.load(path, 3, 'float')
    input = image.scale(input, loadSize[2], loadSize[2])

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = input:size(2)
    local iW = input:size(3)
    
    if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end
    
    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    end
    
    
    if opt.flip == 1 and torch.uniform() > 0.5 then 
      input = image.hflip(input)
    end
    
--    print(input:mean(), input:min(), input:max())
    local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()
    local imA = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0
    local imB = input_lab[{{2,3},{},{}}]:div(110.0)
    local imAB = torch.cat(imA, imB, 1)
    assert(imAB:max()<=1,"A: badly scaled inputs")
    assert(imAB:min()>=-1,"A: badly scaled inputs")
    
    return imAB
end


local function loadImageRandPoint(path)
  -- Load an image, do randomcrop+jitter as usual
  -- imA has 4 channels: L, a, b, mask (indicates which pixels have ground truth ab value)
  -- imB has 2 channels: a, b
    local input = image.load(path, 3, 'float')
    input = image.scale(input, loadSize[2], loadSize[2])

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = input:size(2)
    local iW = input:size(3)
    
    if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end
    
    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    end
    
    
    if opt.flip == 1 and torch.uniform() > 0.5 then 
      input = image.hflip(input)
    end
    
--    print(input:mean(), input:min(), input:max())
    local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()

    local imB = input_lab[{{2,3},{},{}}]:div(110.0)
    -- print('imB', imB:size())

    -- Randomly sample a single point
    local N = -1 + math.ceil(torch.uniform(1e-2, 7))

    local samp_ab = torch.zeros(imB:size())
    -- samp_ab[{{},{a_ind},{b_ind}}] = input_lab[{{2,3},{a_ind},{b_ind}}]
    -- print(samp_ab[{{},{a_min,a_max},{b_min,b_max}}]:size())
    -- print(input_lab[{{2,3},{a_min,a_max},{b_min,b_max}}]:size())
    local mask_ab = torch.zeros(imB[{{1},{},{}}]:size())

    for nn = 1,N do
      local P = 2+math.ceil(torch.uniform(1e-2, 5)) --random half-patch size
      local a_ind = math.ceil(torch.uniform(1e-2, oW))
      local b_ind = math.ceil(torch.uniform(1e-2, oH))

      local a_min = math.max(a_ind-P,1)
      local a_max = math.min(a_ind+P,oW)
      local b_min = math.max(b_ind-P,1)
      local b_max = math.min(b_ind+P,oH)

      samp_ab[{{1},{a_min,a_max},{b_min,b_max}}] = torch.mean(input_lab[{{2},{a_min,a_max},{b_min,b_max}}])
      samp_ab[{{2},{a_min,a_max},{b_min,b_max}}] = torch.mean(input_lab[{{3},{a_min,a_max},{b_min,b_max}}])

      -- Mask for where ground truth color is
      mask_ab[{{},{a_min,a_max},{b_min,b_max}}] = 1
    end

    -- print('a_min',a_min,'a_ind',a_ind,'a_max',a_max)
    -- print('b_min',b_min,'b_ind',b_ind,'b_max',b_max)

    -- samp_ab[{{},{a_min,a_max},{b_min,b_max}}] = input_lab[{{2,3},{a_min,a_max},{b_min,b_max}}]

    -- Lightness image
    local im_L = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0

    -- print('im_L', im_L:size())
    -- print('samp_ab',samp_ab:size())
    -- print('mask_ab',mask_ab:size())

    local imA = torch.cat(im_L, samp_ab ,1)
    -- local imA = torch.cat(im_L, imB ,1) -- just for debugging, add whole color image
    imA = torch.cat(imA, mask_ab, 1)
    -- print('imA 1st', imA:size())
    -- print('imA 2nd', imA:size())

    local imAB = torch.cat(imA, imB, 1)
    -- print('imAB', imAB:size())

    assert(imAB:max()<=1,"A: badly scaled inputs")
    assert(imAB:min()>=-1,"A: badly scaled inputs")
    
    return imAB
end


--local function loadImage

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   local h = input:size(2)
   local w = input:size(3)

   local imA = image.crop(input, 0, 0, w/2, h)
   local imB = image.crop(input, w/2, 0, w, h)
   
   return imA, imB
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   if opt.preprocess == 'regular' then
--     print('process regular')
     local imA, imB = loadImage(path)
     imA, imB = preprocessAandB(imA, imB)
     imAB = torch.cat(imA, imB, 1)
   end
   
   if opt.preprocess == 'colorization' then 
--     print('process colorization')
     imAB = loadImageChannel(path)
   end

   if opt.preprocess == 'color_randpoint' then
     imAB = loadImageRandPoint(path)
   end
--   print('image AB size')
--   print(imAB:size())
   return imAB
end

--------------------------------------
-- trainLoader
print('trainCache', trainCache)
--if paths.filep(trainCache) then
--   print('Loading train metadata from cache')
--   trainLoader = torch.load(trainCache)
--   trainLoader.sampleHookTrain = trainHook
--   trainLoader.loadSize = {input_nc, opt.loadSize, opt.loadSize}
--   trainLoader.sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]}
--   trainLoader.serial_batches = opt.serial_batches
--   trainLoader.split = 100
--else
print('Creating train metadata')
--   print(opt.data)
print('serial batch:, ', opt.serial_batches)
trainLoader = dataLoader{
    paths = {opt.data},
    loadSize = {input_nc, loadSize[2], loadSize[2]},
    sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]},
    split = 100,
    serial_batches = opt.serial_batches, 
    verbose = true
 }
--   print('finish')
--torch.save(trainCache, trainLoader)
--print('saved metadata cache at', trainCache)
trainLoader.sampleHookTrain = trainHook
--end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end