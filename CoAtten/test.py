import lutorpy as lua

require('load')
require('predict')

protos = lod()

img_path = 'vis/demo_img1.jpg'
question1 = 'what is the man doing' 
question2 = 'what is the color of the hat' 

print(pred(img_path, question1, protos))
print(pred(img_path, question2, protos))