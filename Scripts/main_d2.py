from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
import torchvision
import torch
import openpifpaf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from openpifpaf import encoder
import tensorflow as tf
head_names = ['cif','caf']
import datetime

INPUT_SHAPE = (385,385,3)
IMG_SHAPE     = 385
AUGUMENTATION = True
IMAGE_DIR     = '/home/unreal/shared_folder/unrealai/coco17/train2017'
ANN_DIR       = '/home/unreal/shared_folder/unrealai/coco17/annotations/person_keypoints_train2017.json'
BATCH_SIZE    = 8
NUM_WORKERS   = 4
SHUFFLE       = True
PIN_MEMORY    = False


class BatchNorm(Layer):
    def __init__(self, scale=True, center=True):
        super(BatchNorm, self).__init__()        
        self.bn = BatchNormalization(scale=scale, center=center, trainable=True)

    @tf.function
    def call(self, inputs, training=True):
        self.bn.trainable=training
        return self.bn(inputs)    
class CIF(Model):
    def __init__(self):
        super(CIF, self).__init__()
        self.conv  = tf.keras.layers.Conv2D(340, kernel_size=1, strides=(1, 1), padding='valid',  dilation_rate=(1, 1), use_bias=True)
        self.bn = BatchNorm()

    @tf.function
    def pixel_shuffle_tf(self,inputs, scale_factor):
        in_channels, in_height, in_width = inputs.shape

        out_channels = in_channels // (scale_factor * scale_factor)
        out_height = in_height * scale_factor
        out_width = in_width * scale_factor

        if scale_factor >= 1:
            input_view = tf.reshape(inputs,[out_channels,scale_factor, scale_factor, in_height, in_width])
            shuffle_out = tf.transpose(input_view,perm=[0, 3, 1, 4, 2])

        return tf.reshape(shuffle_out,[out_channels, out_height, out_width])  
    @tf.function    
    def call(self, inputs, training=True):
        x = inputs
        
        x = self.conv(x)
        x = self.bn(x)

        x=tf.transpose(x,perm=[0,3,1,2])
        y = None
        batch_size = x.shape[0]
#         batch_size = 1

        for i in range(x.shape[0]):
            if y==None:
                y = tf.expand_dims(self.pixel_shuffle_tf(x[i],2), axis=0)
            else:
                y = tf.concat([y,tf.expand_dims(self.pixel_shuffle_tf(x[i],2), axis=0)],axis=0)

        x = y
        
        x = x[:,:,:-1,:-1]

        classes_x = x[:, :17] 
        
        f_map = classes_x.shape[2]
        classes_x = tf.reshape(classes_x, [batch_size,17, 1, f_map, f_map])
        
        regs_x = x[:, 17:51]
        regs_x = tf.reshape(regs_x, [batch_size,17, 1, 2, f_map, f_map])
        
        regs_logb = x[:, 51:68]
        regs_logb = tf.reshape(regs_logb, [batch_size,17, 1, f_map, f_map])
        
        scales_x = x[:,68:85]
        scales_x = tf.reshape(scales_x, [batch_size,17, 1, f_map, f_map])
                
        return classes_x, regs_x, regs_logb, scales_x
class CAF(Model):
    '''
        Shufflenet Starter Head, made using netron
        structure of the exported onnx model
    '''
    
    
    def __init__(self):
        super(CAF, self).__init__()
        self.conv  = tf.keras.layers.Conv2D(
                                684, kernel_size=1, strides=(1, 1), padding='valid',  dilation_rate=(1, 1), use_bias=True)
        self.bn = BatchNorm()
    
    @tf.function
    def pixel_shuffle_tf(self,inputs, scale_factor):
        in_channels, in_height, in_width = inputs.shape

        out_channels = in_channels // (scale_factor * scale_factor)
        out_height = in_height * scale_factor
        out_width = in_width * scale_factor

        if scale_factor >= 1:
            input_view = tf.reshape(inputs,[out_channels,scale_factor, scale_factor, in_height, in_width])
            shuffle_out = tf.transpose(input_view,perm=[0, 3, 1, 4, 2])

        return tf.reshape(shuffle_out,[out_channels, out_height, out_width])
    
    @tf.function
    def call(self, inputs, training=True):
        x = inputs
        x = self.conv(x)
        x = self.bn(x)
        x=tf.transpose(x,perm=[0,3,1,2])
        
        y = None
        batch_size = x.shape[0]
#         batch_size = 1
        
        for i in range(x.shape[0]):
            if y==None:
                y = tf.expand_dims(self.pixel_shuffle_tf(x[i],2), axis=0)
            else:
                y = tf.concat([y,tf.expand_dims(self.pixel_shuffle_tf(x[i],2), axis=0)],axis=0)
        
        x = y
        
        x = x[:,:,:-1,:-1]
        
        classes_x = x[:, :19] 
        f_map = classes_x.shape[2]
        classes_x = tf.reshape(classes_x, [batch_size,19, 1, f_map, f_map])
#         classes_x = tf.math.sigmoid(classes_x)
        
        regs_x = x[:, 19:95]
        regs_x = tf.reshape(regs_x, [batch_size,19, 2, 2, f_map, f_map])
        
        
        regs_logb = x[:, 95:133]
        regs_logb = tf.reshape(regs_logb, [batch_size,19, 2, f_map, f_map])
        
        scales_x = x[:,133:]
        scales_x = tf.reshape(scales_x, [batch_size,19, 2, f_map, f_map])
                
        return classes_x, regs_x, regs_logb, scales_x    
    
class PIFPAF(Model):
    def __init__(self):
        super(PIFPAF, self).__init__()
#         self.shuffle = Shufflenet2k()
        babli = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet', input_tensor=None, input_shape=INPUT_SHAPE)
        self.shuffle = Model(inputs=babli.input, outputs=babli.get_layer('block6a_expand_activation').output)

#         self.shuffle = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet', input_tensor=None, input_shape=INPUT_SHAPE)
        self.cif = CIF()
        self.caf = CAF()

    @tf.function    
    def call(self, inputs, training=True):
    
        x = self.shuffle(inputs,training)
        print(" SHUFFLE SHAPE ", x.shape)
        x_cif0, x_cif1, x_cif2, x_cif3 = self.cif(x,training)
        x_caf0, x_caf1, x_caf2, x_caf3 = self.caf(x,training)
        return [x_cif0, x_cif1, x_cif2, x_cif3], [x_caf0, x_caf1, x_caf2, x_caf3]
    
    
def custom_loss(outputs, targets, vec):
    x = outputs
    t = targets
    
    def confidence_loss_tf(x_confidence, target_confidence):
        
        
        bce_masks=tf.math.is_nan(target_confidence)
        bce_masks = tf.math.logical_not(bce_masks)
        if not tf.math.reduce_any(bce_masks):
            return tf.convert_to_tensor(np.nan)
        x_confidence = x_confidence[:, :, 0]

        batch_size = x_confidence.shape[0]
        bce_target=tf.boolean_mask(target_confidence, bce_masks)
        bce_weight = 1.0
        x_confidence = tf.boolean_mask(x_confidence, bce_masks)
        bce_weight=tf.zeros_like(bce_target)

        bce_weight=tf.where(bce_target == 1, x_confidence, bce_weight)
        bce_weight=tf.where(bce_target == 0, -x_confidence, bce_weight)
        bce_weight = tf.math.pow((1.0 + tf.math.exp(bce_weight)),-1)

        ce_loss=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=bce_target, 
                                                                      logits=x_confidence)*bce_weight)/(1000*batch_size)
        return ce_loss


    


    def laplace_loss_tf(x1, x2, logb, t1, t2, weight=None):
        norm=tf.norm(tf.stack([x1,x2])-tf.stack([t1,t2]), ord='euclidean', axis=0)
        logb = 3.0 * tf.math.tanh(logb / 3.0)

        losses = 0.694 + logb + norm * tf.math.exp(-logb)
        losses = losses * weight
        return tf.reduce_sum(losses)

    def localization_loss_tf( x_regs, x_logbs, target_regs):
        
        batch_size = target_regs[0].shape[0]
        reg_losses = []
        for i, target_reg in enumerate(target_regs):
            
            target_reg = target_reg
            
            reg_masks=tf.math.is_nan(target_reg[:,:,0])
            reg_masks = tf.math.logical_not(reg_masks)
            if not tf.math.reduce_any(reg_masks):
                reg_losses.append(tf.convert_to_tensor(np.nan))
                continue
                
            x_regs = tf.convert_to_tensor(x_regs)
            target_reg = tf.convert_to_tensor(target_reg)
            x_logbs = tf.convert_to_tensor(x_logbs)

            a,b,c,d,e  =   x_regs[:, :, i, 0],  x_regs[:, :, i, 1],  x_logbs[:, :, i],  target_reg[:, :, 0], target_reg[:, :, 1]
            reg_losses.append(laplace_loss_tf(
                a[reg_masks],
                b[reg_masks],
                c[reg_masks],
                d[reg_masks],
                e[reg_masks],
                weight=0.1,
            ) / (100.0 * batch_size))

        return reg_losses

    def scale_losses_tf(x_scales, target_scales):
        batch_size = x_scales.shape[0]
        return [
#             logl1_loss_tf(
#                 tf.boolean_mask(x_scales[:,:,i], tf.math.logical_not(tf.math.is_nan(target_scale))),
#                 tf.boolean_mask(target_scale, tf.math.logical_not(tf.math.is_nan(target_scale)),
#                 reduction='sum',
#             ) / (100.0 * batch_size)
#             for i, target_scale in enumerate(target_scales)
#         ]
                
                logl1_loss_tf(
                tf.boolean_mask(x_scales[:,:,i], tf.math.logical_not(tf.math.is_nan(target_scale))),
                tf.boolean_mask(target_scale, tf.math.logical_not(tf.math.is_nan(target_scale))),
                reduction='sum',
            ) / (100.0 * batch_size)
            for i, target_scale in enumerate(target_scales)
        ]
    
    
    def logl1_loss_tf(logx, t, **kwargs):
        """Swap in replacement for functional.l1_loss."""
        logt = tf.math.log(t)
        res = tf.math.abs(logx-logt)
        res = tf.reduce_sum(res)
        return res


    def margin_losses_tf( x_regs, target_regs, *, target_confidence):
        return []


    x_confidence, x_regs, x_logbs, x_scales = x

    assert len(t) == 3 or 5
    running_t = iter(t)
    target_confidence = next(running_t)
    target_regs = [next(running_t) for _ in range(vec)]
    target_scales = [next(running_t) for _ in range(vec)]
    
    #print(" confidence shape", x_confidence.shape )

    ce_loss = confidence_loss_tf(x_confidence, target_confidence)
    reg_losses = localization_loss_tf(x_regs, x_logbs, target_regs)
    scale_losses = scale_losses_tf(x_scales, target_scales)
    margin_losses = margin_losses_tf(x_regs, target_regs,target_confidence=target_confidence)
    return [ce_loss] + reg_losses + scale_losses + margin_losses
def create_input_batch(data):
    da = data.numpy()
    
    #da = (da - np.min(da))/np.ptp(da)
    da = np.transpose(da, (0,2,3,1))
#     da = da[:,,:-1,:]

    return tf.convert_to_tensor(da)
def calculate_loss(res, target_batch):
    #print(" RES SHAPE", res[0][1].shape)
    cif_loss = custom_loss(res[0], target_batch[0], 1)
    caf_loss = custom_loss(res[1], target_batch[1], 2)
    flat_head_losses = []
    flat_head_losses.extend(cif_loss)
    flat_head_losses.extend(caf_loss)
    lambdas = [1.0,1.0,0.2,1.0 ,1.0, 1.0, 0.2,0.2]
#     lambdas = [1.0,0.0,0.0,1.0 ,0.0, 0.0, 0.0,0.0]
#     lambdas=[1.0,1.0]
    loss_values=[]
    
    
    
    for lam, l in zip(lambdas, flat_head_losses):
        r=tf.math.multiply(lam,l)
        if not tf.math.is_nan(r):
            loss_values.append(r)
    a=tf.constant(np.nan)
    total_loss = tf.cond(tf.reduce_sum(loss_values)==0  ,lambda: a, lambda: tf.reduce_sum(loss_values))
    return total_loss,flat_head_losses
def calculate_loss(res, target_batch):
    #print(" RES SHAPE", res[0][1].shape)
    cif_loss = custom_loss(res[0], target_batch[0], 1)
    caf_loss = custom_loss(res[1], target_batch[1], 2)
    flat_head_losses = []
    flat_head_losses.extend(cif_loss)
    flat_head_losses.extend(caf_loss)
    lambdas = [1.0,1.0,0.2,1.0 ,1.0, 1.0, 0.2,0.2]
#     lambdas = [1.0,0.0,0.0,1.0 ,0.0, 0.0, 0.0,0.0]
#     lambdas=[1.0,1.0]
    loss_values=[]
    
    
    
    for lam, l in zip(lambdas, flat_head_losses):
        r=tf.math.multiply(lam,l)
        if not tf.math.is_nan(r):
            loss_values.append(r)
    a=tf.constant(np.nan)
    total_loss = tf.cond(tf.reduce_sum(loss_values)==0  ,lambda: a, lambda: tf.reduce_sum(loss_values))
    return total_loss,flat_head_losses


print(" LOADING DATASET ")



def create_input_batch(data):
    da = data.numpy()
    
    #da = (da - np.min(da))/np.ptp(da)
    da = np.transpose(da, (0,2,3,1))
#     da = da[:,,:-1,:]

    return tf.convert_to_tensor(da)

base_vision = torchvision.models.ShuffleNetV2([4, 8, 4], [24, 348, 696, 1392, 1392])
blocks = [base_vision.conv1, 
        base_vision.stage2, base_vision.stage3, base_vision.stage4, base_vision.conv5,]
basenet = openpifpaf.network.basenetworks.BaseNetwork(
        torch.nn.Sequential(*blocks),
        "shufflenetv2k16w",
        stride=16,
        out_features=1392,
    )
head_metas = openpifpaf.datasets.headmeta.factory(head_names)
headnets = [openpifpaf.network.heads.CompositeFieldFused(h, basenet.out_features) for h in head_metas]
net_cpu = openpifpaf.network.nets.Shell(basenet, headnets)
preprocess = openpifpaf.datasets.factory.train_cocokp_preprocess_factory(
        square_edge=IMG_SHAPE,
        augmentation=AUGUMENTATION,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0)
original_transforms = encoder.factory(net_cpu.head_nets, net_cpu.base_net.stride)
train_data = openpifpaf.datasets.Coco(
         image_dir = IMAGE_DIR,
         ann_file  = ANN_DIR,
         preprocess=preprocess,
         target_transforms=original_transforms,
         n_images=None,
         image_filter='keypoint-annotations',
         category_ids=[1],
    )
def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=SHUFFLE,
                                           pin_memory=PIN_MEMORY,
                                           num_workers=NUM_WORKERS,
                                           drop_last=True,
                                           collate_fn=collate_images_targets_meta)
print("Data Loaded")




print(" Setting Optimizer ")

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.8,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


print(" Optimizer Set ")


print("Creating Model")

model = PIFPAF()
tert = np.random.rand(16,IMG_SHAPE,IMG_SHAPE,3).astype(np.float32)
bert = model(tert)
# model.load_weights('/home/unreal/shared_folder/model/new_mobile/epoch_35/atom3001.h5')
print(" Creating Logs ")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("current_time", current_time)
train_log_dir = 'logs/Efficient/' + str(current_time) + '/train'
print(" train_log_dir ",train_log_dir) 
test_log_dir =  'logs/Efficient/' + str(current_time) + '/test'

print(" Logs Created ")

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

print("Summary Created")


# model.load_weights("/home/unreal/shared_folder/model/mobile_final/epoch_62/atom3001.h5")

kk=0
arr=[]
prev_weights= []
epochs = 100
loss_value = 0.0
tt=0
import time
#@tf.function
def train_step(data, target_batch):
    with tf.GradientTape() as tape:
        res = model(tf.convert_to_tensor(data), training=True)  # Logits for this minibatch  
        loss_value,flat_head_losses = calculate_loss(res,tt)
    grads = tape.gradient(loss_value, model.trainable_variables)
    a=[]
    b=[]
    wei=model.trainable_variables
    for i in range(len(grads)):
        x=tf.math.reduce_sum(grads[i])
        y=tf.math.reduce_sum(wei[i])
        a.append(grads[i])
        b.append(wei[i])
        
    
    optimizer.apply_gradients(zip(a, b))
    return loss_value,flat_head_losses,res,grads

arr=['cif_confidence','cif_local','cif_scale', 'cif_margin','caf_confidence','caf_local','caf_scale', 'caf_margin']
print(" Lenth of dataloader " , len(train_loader))

print(" Training Started ")

for epoch in range(0,epochs):
    step=0
    it = iter(train_loader)
    while(True):
        try:
            (data, target, meta) = next(it)
            step+=1
        except Exception as e:
            step+=1
            if step>=len(train_loader):
                break
            continue
        if step>=len(train_loader):
            break
        
        target_batch = target
        data = create_input_batch(data)
        ref_start = time.time()
        tt=[]
        for i in target_batch:
            aa=[]
            for j in i:
                aa.append(tf.convert_to_tensor(j))
            tt.append(aa)
        total_loss,flat_head_losses,res,grads=train_step(data, tt)        
        
        print(" Epoch - ",epoch, "Step  " , step," total Loss ", total_loss.numpy(), time.time()-ref_start)
    
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', total_loss, step=(step+(epoch*len(train_loader))))
            for i in range(8):
                tf.summary.scalar(str(arr[i]), flat_head_losses[i], step=(step+(epoch*len(train_loader))))

        if step%1000==0:
            print("Hello EfficientNet "+str(step))
            if os.path.exists("/home/unreal/shared_folder/POSE/EfficentD2/epoch_"+str(epoch)):
                model.save_weights("/home/unreal/shared_folder/POSE/EfficentD2/epoch_"+str(epoch)+"/"+'atom{}.h5'.format(step + 1))
            else:
                os.mkdir("/home/unreal/shared_folder/POSE/EfficentD2/epoch_"+str(epoch))
                model.save_weights("/home/unreal/shared_folder/POSE/EfficentD2/epoch_"+str(epoch)+"/"+'atom{}.h5'.format(step + 1))



