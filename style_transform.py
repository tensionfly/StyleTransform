from mxnet import image
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from time import time
from mxnet import autograd
import mxnet as mx
import cv2
#data
style_img_path='st.jpg'
content_img_path='whu.jpg'

style_img=image.imread(style_img_path)
content_img=image.imread(content_img_path)

#model
rgb_mean=nd.array([0.485,0.456,0.406])
rgb_std=nd.array([0.229,0.224,0.225])

def preprocess(img,image_shape):
    img=image.imresize(img,*image_shape)
    img=(img.astype('float32')/255-rgb_mean)/rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img=img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std+rgb_mean).clip(0,1)

def get_net(pretrained_net,content_layers,style_layers):
    net=nn.Sequential()
    for i in range(max(content_layers+style_layers)+1):
        net.add(pretrained_net.features[i])
    return net

def extract_features(x,net,content_layers,style_layers):
    contents=[]
    styles=[]
    for i in range(len(net)):
        x=net[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents,styles

def content_loss(yhat,y):
    return (yhat-y).square().mean()

def gram(x):
    c=x.shape[1]
    n=x.size/x.shape[1]
    y=x.reshape((c,int(n)))
    return nd.dot(y,y.T)/n

def style_loss(yhat,gram_y):
    return (gram(yhat)-gram_y).square().mean()

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:]-yhat[:,:,:-1,:]).abs().mean()+
                (yhat[:,:,:,1:]-yhat[:,:,:,:-1]).abs().mean())

def sum_loss(loss,pred,truths,weights):
    return nd.add_n(*[w*loss(yhat,y) for w, yhat, y in zip(
        weights,pred,truths)])

def get_contents(net,image_shape,content_layers,style_layers):
    content_x=preprocess(content_img,image_shape)
    content_y,_=extract_features(content_x,net,content_layers,style_layers)
    return content_x,content_y

def get_styles(net,image_shape,content_layers,style_layers):
    style_x=preprocess(style_img,image_shape)
    _,style_y=extract_features(style_x,net,content_layers,style_layers)
    style_y=[gram(y) for y in style_y]
    return style_x,style_y

pretrained_net=models.vgg16(pretrained=True)
print(pretrained_net)

style_layers=[0,5,10,17,24]
content_layers=[21]
net=get_net(pretrained_net,content_layers,style_layers)

channels=[net[l].weight.shape[0] for l in style_layers]
style_weights=[1e4/n**2 for n in channels]#1e4
content_weights=[1]
tv_weight=10

def train(x,max_epochs,lr,lr_decay_epoch=200):
    tic=time()
    for i in range(max_epochs):
        with autograd.record():
            content_py,style_py=extract_features(x,net,content_layers,style_layers)
            
            content_L=sum_loss(content_loss,content_py,content_y,content_weights)
            style_L=sum_loss(style_loss,style_py,style_y,style_weights)
            tv_L=tv_weight*tv_loss(x)

            loss=style_L+content_L+tv_L
        loss.backward()
        x.grad[:]/=x.grad.abs().mean()+1e-8
        x[:]-=lr*x.grad
        nd.waitall()

        if i and i%20==0:
            print('epoch %3d, content_loss: %.2f, style_loss: %.2f,tv_loss: %.2f, \
                    time: %.1f sec'%(i,content_L.asscalar(),style_L.asscalar(),tv_L.asscalar(),time()-tic))
            tic=time()

        if i and i%lr_decay_epoch==0:
            lr*=0.1
            print('change lr to ',lr)
    return x

#输出图尺寸与内容图尺寸相同，风格图尺寸与内容图不需要相同
ctx=mx.cpu()

image_shape=(300,400)
content_x,content_y=get_contents(net,image_shape,content_layers,style_layers)
style_x,style_y=get_styles(net,image_shape,content_layers,style_layers)

net.collect_params().reset_ctx(ctx)

x=content_x.copyto(ctx)
x.attach_grad()
y=train(x,1000,0.1)

cv2.imwrite('style_transform_content_init_'+style_img_path+'.jpg',postprocess(y).asnumpy()*255)
print("style_transform_content_init 保存成功！")

image_shape=(1277,850)
content_x,content_y=get_contents(net,image_shape,content_layers,style_layers)
style_x,style_y=get_styles(net,image_shape,content_layers,style_layers)

x=preprocess(postprocess(y)*255,image_shape).copyto(ctx)
x.attach_grad()
z=train(x,500,0.1,100)

cv2.imwrite('style_transform_learn_init_'+style_img_path+'.jpg',postprocess(z).asnumpy()*255)
print("style_transform_learn_init 保存成功！")