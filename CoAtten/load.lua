require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoader'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten'
require 'misc.cnnModel'
require 'misc.optim_updates'
utils = require 'misc.utils'
require 'xlua'


function load()
	opt = {}

	opt.vqa_model = 'model/vqa_model/model_alternating_train-val_vgg.t7'
	opt.cnn_proto = 'image_model/VGG_ILSVRC_19_layers_deploy.prototxt'
	opt.cnn_model = 'image_model/VGG_ILSVRC_19_layers.caffemodel'
	opt.json_file = 'data/vqa_data_prepro_all.json'
	opt.backend = 'cudnn'
	opt.gpuid = 1
	if opt.gpuid >= 0 then
	  require 'cutorch'
	  require 'cunn'
	  if opt.backend == 'cudnn' then 
	  require 'cudnn' 
	  end
	  --cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
	end

	local loaded_checkpoint = torch.load(opt.vqa_model)
	local lmOpt = loaded_checkpoint.lmOpt

	lmOpt.hidden_size = 512
	lmOpt.feature_type = 'VGG'
	lmOpt.atten_type = 'Alternating'
	cnnOpt = {}
	cnnOpt.cnn_proto = opt.cnn_proto
	cnnOpt.cnn_model = opt.cnn_model
	cnnOpt.backend = opt.backend
	cnnOpt.input_size_image = 512
	cnnOpt.output_size = 512
	cnnOpt.h = 14
	cnnOpt.w = 14
	cnnOpt.layer_num = 37

	-- load the vocabulary and answers.

	local json_file = utils.read_json(opt.json_file)
	ix_to_word = json_file.ix_to_word
	ix_to_ans = json_file.ix_to_ans

	word_to_ix = {}
	for ix, word in pairs(ix_to_word) do
	    word_to_ix[word]=ix
	end

	-- load the model
	protos = {}
	protos.word = nn.word_level(lmOpt)
	protos.phrase = nn.phrase_level(lmOpt)
	protos.ques = nn.ques_level(lmOpt)

	protos.atten = nn.recursive_atten()
	protos.crit = nn.CrossEntropyCriterion()
	protos.cnn = nn.cnnModel(cnnOpt)

	if opt.gpuid >= 0 then
	  for k,v in pairs(protos) do v:cuda() end
	end

	cparams, grad_cparams = protos.cnn:getParameters()
	wparams, grad_wparams = protos.word:getParameters()
	pparams, grad_pparams = protos.phrase:getParameters()
	qparams, grad_qparams = protos.ques:getParameters()
	aparams, grad_aparams = protos.atten:getParameters()

	print('Load the weight...')
	wparams:copy(loaded_checkpoint.wparams)
	pparams:copy(loaded_checkpoint.pparams)
	qparams:copy(loaded_checkpoint.qparams)
	aparams:copy(loaded_checkpoint.aparams)

	print('total number of parameters in cnn_model: ', cparams:nElement())
	assert(cparams:nElement() == grad_cparams:nElement())

	print('total number of parameters in word_level: ', wparams:nElement())
	assert(wparams:nElement() == grad_wparams:nElement())

	print('total number of parameters in phrase_level: ', pparams:nElement())
	assert(pparams:nElement() == grad_pparams:nElement())

	print('total number of parameters in ques_level: ', qparams:nElement())
	assert(qparams:nElement() == grad_qparams:nElement())
	protos.ques:shareClones()

	print('total number of parameters in recursive_attention: ', aparams:nElement())
	assert(aparams:nElement() == grad_aparams:nElement())

	return protos
end