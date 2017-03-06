require 'torch'
require 'image'
utils = require 'misc.utils'

--img_path = 'vis/demo_img1.jpg'
--question = 'what is the man doing' 
function predict(img_path, question, protos)
  -- load the image
  local img = image.load(img_path)
  -- scale the image
  img = image.scale(img,448,448)
  --itorch.image(img)
  img = img:view(1,img:size(1),img:size(2),img:size(3))

  -- parse and encode the question (in a simple way).
  local ques_encode = torch.IntTensor(26):zero()

  local count = 1
  for word in string.gmatch(question, "%S+") do
      ques_encode[count] = word_to_ix[word] or word_to_ix['UNK']
      count = count + 1
  end
  ques_encode = ques_encode:view(1,ques_encode:size(1))
  -- ques_encode is a vector of the words indices in some dictionary

  -- doing the prediction

  local image_raw = utils.prepro(img, false)
  image_raw = image_raw:cuda()
  ques_encode = ques_encode:cuda()

  local image_feat = protos.cnn:forward(image_raw)
  local ques_len = torch.Tensor(1,1):cuda()
  ques_len[1] = count-1

  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({ques_encode, image_feat}))
  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, ques_len, img_feat, mask}))
  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
  local out_feat = protos.atten:forward(feature_ensemble)

  local tmp,pred=torch.max(out_feat,2)
  local ans = ix_to_ans[tostring(pred[1][1])]

  return ans

end