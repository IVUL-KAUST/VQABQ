require 'load'
require 'predict'

protos = lod()

-- specify the image and the question.
local img_path = 'vis/demo_img1.jpg'
local question = 'what is the man doing' 

print(pred(img_path, question, protos))
print(pred(img_path, 'what is the color of the hat', protos))