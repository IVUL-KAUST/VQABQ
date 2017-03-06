import lutorpy as lua

require('load')
require('predict')

protos = load()

img_path = 'vis/demo_img1.jpg'
question1 = 'what is the man doing' 
question2 = 'what is the color of the hat' 

print(predict(img_path, question1, protos))
print(predict(img_path, question2, protos))