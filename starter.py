""" Northwestern MSiA 490 Deep learning 
    Assignment #2 starter code, Spring 2018
    This code demonstrates the use of transfer learning to speed up
    the training process for a convolutional neural network.
    Notes:
        - Heatmaps may appear black in the first few epochs. Run to ~500 to get decent maps.
        - Watch out for memory usage. Big jobs get terminated by the system when memory is low.
        - The native image size is 224x224 for VGG, resize/crop your images to match
        - Filter visualization is slow, change cfg.vizfilt_timeout for more speed or accuracy
        - Be sure to rename/delete the basepath when changing model parameters, e.g. layers or random labels
        - Change cfg.spercls to use larger/smaller batches
        - Filter visualization is not compatible with batch normalization
        - 3 step process: (1) Overfit training set (min size), (2) Fit random labels (max size), (3) Regularize for test/validation set
"""
import os, time, numpy as np, scipy, random, tqdm, pandas as pd, socket
import matplotlib.pyplot as plt
from keras.optimizers import Adam#, SGD, RMSprop, Adagrad
np.seterr(divide='ignore', invalid='ignore')
#%%
class ProjectConfig:
    basepath      = "assignment2-data"
    imsize        = (32,) * 2 # n x n square images, VGG default is 224x224
    tsize         = imsize + (3,)
    trainfolder   = os.path.join(basepath, 'train-%s' % str(tsize))
    testfolder    = os.path.join(basepath, 'test-%s' % str(tsize)) # Set test = train to overfit train data

# Model settings    
cfg = ProjectConfig()
cfg.vgglayers     = 2        # Number of VGG layers to create, 0-5 layers
cfg.xferlearning  = -1       # Enable transfer learning up to layer n (max 12, -1 = off)
cfg.freeze_conv   = False    # Freeze convolutional layers
cfg.fclayersize   = 128      # Size of fully connected (FC) layers
cfg.fclayers      = 1        # Number of FC layers
cfg.fcdropout     = 0.0      # Dropout regularization factor for FC layers
cfg.l1            = 0.0      # L1 regularization for FC
cfg.l2            = 0.0      # L2 regularization for FC
cfg.randomlbls    = False    # Random labels (remember to rename/delete existing data folder)
 
# Optimizer settings
optimizer = Adam()   
cfg.batch_size, cfg.nb_epoch = 32, 10000 # Change for early stopping regularization
cfg.batchnorm     = False    # Batch normalization (incompatible with filter viz)
cfg.saveloadmodel = True     # Save/load models to reduce training time
cfg.frac          = 0.25     # Fraction of data to load

# Visualization settings
cfg.hsv           = False    # Convert images to Hue/Saturation/Value to be more robust to colors
cfg.vizfilt_timeout = 3      # Decrease for speed, increase for better viz. 0 = off.

# Model checkpointing/stats
cfg.modelarch     = 'vgg%d-fcl%d-fcs%d-%s-%s' % (cfg.vgglayers, cfg.fclayers, cfg.fclayersize, 'hsv' if cfg.hsv else 'rgb', socket.gethostname())
cfg.modelid       = os.path.join(cfg.basepath, 'model-%s.h5' % cfg.modelarch)
cfg.trainstats    = os.path.join(cfg.basepath, 'trainstats-%s.csv' % cfg.modelarch)        

#%% Image data augmentation
cfg.adapthist     = True     # Adapts/standardizes image histogram     
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,               # set input mean to 0 over the dataset
    samplewise_center=False,                # set each sample mean to 0
    featurewise_std_normalization=False,    # divide inputs by std of the dataset
    samplewise_std_normalization=False,     # divide each input by its std
    zca_whitening=False,                    # apply ZCA whitening
    rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.0,                  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.0,                 # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,                  # randomly flip images
    vertical_flip=False)                    # randomly flip images
#%% ------ CPU/GPU memory fix -------
import tensorflow as tf, keras.backend.tensorflow_backend as ktf
def get_session(gpu_fraction=0.45):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

import psutil, gc
def check_memusage():
    process = psutil.Process(os.getpid())
    pctused = 100*process.memory_info().rss / psutil.virtual_memory().total
    if pctused > 50:
        print('**** High memory usage detected***')
        print('Please check your code for memory leaks. Percent memory used: ', pctused)
#%% Create demo data
def makedata(basepath):
    if os.path.exists(basepath): return
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    if cfg.randomlbls: y_train = np.random.randint(0, len(np.unique(y_train)), y_train.shape)
    obj_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
    for (X_data, y_data, bp) in [(X_train, y_train, cfg.trainfolder), (X_test, y_test, cfg.testfolder)]:
        for c in obj_classes: os.makedirs(os.path.join(bp, c), exist_ok=True)
        for i, (im, lbl) in tqdm.tqdm(enumerate(zip(X_data, y_data)), desc='Making data folder', total=len(y_data), mininterval=2):
            pn = os.path.join(bp, obj_classes[int(lbl)], "%d.png" % i)
            if not os.path.exists(pn): scipy.misc.imsave(pn, scipy.misc.imresize(im, cfg.imsize, interp='bicubic'))
makedata(cfg.basepath) # Comment out this line to use your own data
#%% Dataset loader
class DataSet:
    def __init__(self):
        self.columns = ['Type', 'Path', 'Class', 'Fname', 'xTrain', 'xCorrect', 'xWrong', 'Mean', 'Var']
        self.configfile = os.path.join(cfg.basepath, 'dataset.csv')
        self.data = pd.DataFrame(columns=self.columns)       
        self.imgcache = {}
        if os.path.exists(self.configfile): self.data = pd.read_csv(self.configfile)
        self.scan()
        
    def scan(self):
        to_insert = []
        for t, d in [('Train', cfg.trainfolder), ('Test', cfg.testfolder)]:
            for root, dir, files in tqdm.tqdm(os.walk(d), mininterval=2, desc='Scanning data folder...'):
                for f in files:
                    cls = os.path.split(root)[-1]
                    path = os.path.join(root, f)
                    if path in self.data['Path']: continue # Ignore files that we've already seen
                    if f[0] == '.': continue               # Ignore hidden files  
                    if '.png' not in f.lower(): continue   # Ignore non-PNG files
                    row = (t, path, cls, f, 0, 0, 0, np.nan, np.nan)
                    to_insert.append(row)
        newdata   = pd.DataFrame(data=to_insert, columns=self.columns)
        self.data = pd.concat([self.data, newdata])
        self.data = self.data.drop_duplicates(['Path'])
        self.data.to_csv(self.configfile, index=False)
        return self.data

    def load_image(self, fn):        
        if fn in self.imgcache: return self.imgcache[fn]        
        import skimage.color, skimage.exposure, skimage.io, skimage.transform    
        
        try:
            im = skimage.io.imread(fn)
            im = skimage.transform.resize(im, cfg.imsize, order=3)
            if cfg.adapthist: im = skimage.exposure.equalize_adapthist(im)
            if cfg.hsv:       im = skimage.color.rgb2hsv(im)
            im = skimage.img_as_ubyte(im)
            self.imgcache[fn] = im
            if psutil.virtual_memory().percent > 95:
                print("Image cache full...")
                for it in random.sample(self.imgcache.keys(), min(20, len(self.imgcache))):
                    del self.imgcache[it]
        except OSError:
            print("Warning: corrupt file %s" % fn)
            return None
        return im
    
    def load_dataset(self, population, timeout):
        import warnings
        tm = time.time()
        xdata, ydata = [], []
        self.obj_classes = sorted(self.data['Class'].unique())
        try:
            for c, o in tqdm.tqdm(zip(population['Class'], population['Path']), desc='Loading samples', total=len(population), mininterval=2):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")                
                    im = self.load_image(o)
                if time.time() - tm > timeout: break
                if im is None: continue
                xdata.append(im)
                clsid = self.obj_classes.index(c)
                ydata.append(clsid)
        except KeyboardInterrupt: pass
        return xdata, ydata

    def load_data(self, basepath, dtype, frac=1.0, timeout=999999):
        """Loads image data using folder names as class names
           Beware: make sure all images are the same size, or resize them manually"""
        print("Cache size:", len(self.imgcache))
            
        assert(dtype in ['Train', 'Test'])
        population = self.data[self.data['Type'] == dtype]
        population = population.sample(min(int(frac*len(population)), len(population)))
        
        xdata, ydata = self.load_dataset(population, timeout)
        
        xdata  = np.array(xdata, dtype='float32')
        xdata -= xdata.min()
        xdata /= xdata.max()
        return xdata, np.array(ydata, dtype='float32'), self.obj_classes
    
dataset = DataSet()

# Load test data
from keras.utils import np_utils
X_test, y_test, obj_classes = dataset.load_data(cfg.testfolder, 'Test')
class_to_idx = dict([(y, x) for (x,y) in enumerate(obj_classes)])
img_rows, img_cols, img_channels = X_test.shape[1:]
Y_test  = np_utils.to_categorical(y_test, len(obj_classes))

#%% VGG net definition
# VGG net definition starts here. Change the vgglayers to set how many layers to transfer
from keras.models import Model
from keras.regularizers import l1_l2
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

img_input = Input(shape=cfg.tsize)
if cfg.vgglayers == 0: x = img_input 
if cfg.vgglayers >= 1: # Block 1
    x = LeakyReLU(0.2)(Convolution2D(64, (3, 3), padding='same', name='block1_conv1')(img_input))
    x = LeakyReLU(0.2)(Convolution2D(64, (3, 3), padding='same', name='block1_conv2')(x))
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    if cfg.batchnorm: x = BatchNormalization()(x)
if cfg.vgglayers >= 2: # Block 2
    x = LeakyReLU(0.2)(Convolution2D(128, (3, 3), padding='same', name='block2_conv1')(x))
    x = LeakyReLU(0.2)(Convolution2D(128, (3, 3), padding='same', name='block2_conv2')(x))
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    if cfg.batchnorm: x = BatchNormalization()(x)
if cfg.vgglayers >= 3: # Block 3
    x = LeakyReLU(0.2)(Convolution2D(256, (3, 3), padding='same', name='block3_conv1')(x))
    x = LeakyReLU(0.2)(Convolution2D(256, (3, 3), padding='same', name='block3_conv2')(x))
    x = LeakyReLU(0.2)(Convolution2D(256, (3, 3), padding='same', name='block3_conv3')(x))
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    if cfg.batchnorm: x = BatchNormalization()(x)
if cfg.vgglayers >= 4: # Block 4
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block4_conv1')(x))
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block4_conv2')(x))
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block4_conv3')(x))
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    if cfg.batchnorm: x = BatchNormalization()(x)
if cfg.vgglayers >= 5: # Block 5
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block5_conv1')(x))
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block5_conv2')(x))
    x = LeakyReLU(0.2)(Convolution2D(512, (3, 3), padding='same', name='block5_conv3')(x))
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if cfg.batchnorm: x = BatchNormalization()(x)
    
x = Flatten(name='flatten')(x)
for i in range(cfg.fclayers): x = Dense(cfg.fclayersize, activation=LeakyReLU(0.2), kernel_regularizer=l1_l2(cfg.l1, cfg.l2))(x)
x = Dense(len(obj_classes), activation='softmax', name='predictions')(x)

inputs = img_input
model = Model(inputs, x, name='vgg16')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
#%% Transfer weights
from keras.applications import vgg16
import keras.layers.convolutional
vgg16model = vgg16.VGG16(include_top=False)
modelconv = [l for l in model.layers if type(l) == keras.layers.convolutional.Conv2D]
vgg16conv = [l for l in vgg16model.layers if type(l) == keras.layers.convolutional.Conv2D]

for i, l in enumerate(modelconv):
    if i > cfg.xferlearning: continue # Transfer only first n layers
    print('**** Transferring layer %d: %s from VGG ****' % (i, l))
    weights = vgg16conv[i].get_weights()
    modelconv[i].set_weights(weights)
    if cfg.freeze_conv: l.trainable = False
#%% Visualization code
def viz_losses(stats): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    epoch = len(stats)
    fig.suptitle("Training vgglayers=%d, fclayers=%d, fcsize=%d, epoch=%d" % (cfg.vgglayers, cfg.fclayers, cfg.fclayersize, epoch))
    ax1.plot(stats['Train loss'].values, label='Train loss', color='blue')
    ax1.plot(stats['Test loss'].values, label='Test loss', color='green')
    ax1.set_yscale('log')
    ax2.plot(stats['Accuracy'].values, label='Test accuracies', color='red')
    ax2.plot(stats['Train accuracy'].values, label='Train accuracies', color='blue')
    ax2.axhline(1.0/len(obj_classes), linestyle='dashed', color='gray')
    chance = dataset.data['Class'].value_counts().max() / dataset.data['Class'].value_counts().sum()
    ax2.text(0, chance, 'Chance')
    ax2.axhline(np.max(stats['Accuracy']), linestyle='dashed', color='red')
    ax2.text(0, np.max(stats['Accuracy']), 'Best')    
    ax2.set_ylim([0, 1])
    ax2.set_title('Accuracy: %0.2f%%' % (100.0*stats['Accuracy'].values[-1]))     
    ax1.legend(), ax2.legend()
    plt.savefig(os.path.join(cfg.basepath, 'loss-%s.png' % cfg.modelarch))
    plt.show()    
    plt.close()
#%% Explanations
import skimage.exposure, skimage.filters
from skimage.color import gray2rgb
from keras import backend as K

def hide_axes(ax): ax.set_xticks([]), ax.set_yticks([])
class Heatmap:
    def __init__(self, model, obj_classes):
        self.obj_classes = obj_classes
        self.nclasses    = len(obj_classes)
        self.model       = model
    
    def make_masks(self, im, n=8, maskval=0.1):
        masks = []
        xwidth, ywidth = int(np.ceil(im.shape[0]/n)), int(np.ceil(im.shape[1]/n))
        for i in range(n):
            for j in range(n):
                mask = np.ones(im.shape[:2])
                mask[(i*xwidth):((i+1)*xwidth), (j*ywidth):((j+1)*ywidth)] = maskval
                mask = skimage.filters.gaussian(mask, 1) # Change this for local mask smoothing
                masks.append(mask)
        return np.array(masks)

    def get_slice_masks(self, im, n_segments=16, blur=0.03):
        from skimage.segmentation import slic
        segments = slic(im, n_segments=n_segments, sigma=5)
        masks = []
        # loop over the unique segment values
        for (i, segVal) in enumerate(np.unique(segments)):
            # construct a mask for the segment
            mask = np.zeros(im.shape[:2], dtype="float32")
            mask[segments == segVal] = 1
            mask = skimage.filters.gaussian(mask, im.shape[1]*blur) # Change this for local mask smoothing            
            masks.append(mask)
        return np.array(masks), segments
        
    def explain_prediction_heatmap(self, im, actual):
        import skimage.color
        from skimage.segmentation import mark_boundaries
        def hsv(im): return skimage.color.hsv2rgb(im) if cfg.hsv else im
        plt.imshow(hsv(im), interpolation='bilinear'), plt.xticks([]), plt.yticks([]), plt.title('Full image'), plt.show(), plt.close()
        masks = np.concatenate([self.make_masks(im, n=i) for i in (9, 7, 5, 3, 2)])
        #masks, segments = self.get_slice_masks(im)        
        masknorm = masks.sum(axis=0)
        heatmaps = np.zeros((self.nclasses,) + im.shape[:2])
        for m in masks:
            prediction = self.model.predict(np.expand_dims(im*gray2rgb(m), 0))
            for c in range(self.nclasses):
                heatmaps[c] += (prediction[0][c]*m)
        for h in heatmaps: h = h / masknorm
        fig, axes = plt.subplots(2, self.nclasses + 1, figsize=(20, 5))
        #axes[0,0].imshow(hsv(im)), axes[1,0].imshow(mark_boundaries(im, segments))        
        axes[0,0].imshow(hsv(im)), axes[1,0].imshow(im)        
        
        axes[0,0].set_title(actual)
        axes[1,0].set_title('HSV' if cfg.hsv else 'RGB')        
        hide_axes(axes[0,0]), hide_axes(axes[1,0])       
        predictions = np.sum(heatmaps, axis=(1,2,))
        predictions /= predictions.max()
        for n, i in enumerate(np.argsort(predictions)[::-1][:self.nclasses]):
            h = ((255 * heatmaps[i])/heatmaps[i].max()).astype('uint16')
            h = skimage.exposure.equalize_adapthist(h)
            h = skimage.filters.gaussian(h, 1) # Change this for global mask smoothing
            axes[0, n+1].imshow(gray2rgb(h))
            axes[1, n+1].imshow(gray2rgb(h) * hsv(im) * (0.5 + 0.5*predictions[i]))  
            hide_axes(axes[0, n+1]), hide_axes(axes[1, n+1])        
            axes[0, n+1].set_title(self.obj_classes[i] + ': %0.1f%%' % (100*predictions[i]/predictions.sum()))
        fig.tight_layout()
        plt.savefig(os.path.join(cfg.basepath, 'heatmap-%05d.png') % np.random.randint(0, 99999))
        plt.show()
        plt.close()
        return heatmaps

class Viz_filters:
    def __init__(self, model, img_input, img_rows, img_cols):
        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])
        self.img_input  = img_input
        self.img_rows   = img_rows
        self.img_cols   = img_cols

    def deprocess_image(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x = x*0.1 + 0.5
        x = np.clip(x, 0, 1) * 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    
    def viz_filter_max(self, layer_name, filter_index=0, max_steps=9999, timeout=3):
        layer_output = self.layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        
        grads = K.gradients(loss, self.img_input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([self.img_input], [loss, grads])
        step = 1e-0
        input_img_data = np.random.random((1, self.img_rows, self.img_cols, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128
    
        tm = time.time()
        for i in tqdm.tqdm(range(max_steps), desc='Filter visualization', mininterval=2):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            if time.time() - tm > timeout:
                plt.text(0.1, 0.1, "Filter viz timeout", color='red')
                break
        img = input_img_data[0]
        img = self.deprocess_image(img)
        fig = plt.imshow(img)
        hide_axes(fig.axes)
        return layer_output
    
    def viz_filters(self, nbfilters=3):
        print("Visualizing filters (CTRL-C to cancel)")
        try: 
            for layer_name in sorted(self.layer_dict.keys()):
                if not hasattr(self.layer_dict[layer_name], 'filters'): continue
                nfilters = self.layer_dict[layer_name].filters
                print("Layer", layer_name, "has", nfilters, "filters")
                plt.subplots(1, nbfilters)
                for j in range(nbfilters):
                    plt.subplot(1, nbfilters, j + 1)
                    self.viz_filter_max(layer_name, random.randint(0, nfilters-1), timeout=cfg.vizfilt_timeout)
                plt.savefig(os.path.join(cfg.basepath, 'filters-%s-%s.png' % (cfg.modelarch, layer_name))) 
                plt.show()                
                plt.close()
        except KeyboardInterrupt: return

class Explainer:
    def __init__(self, model, obj_classes, class_to_idx):
        self.model              = model
        self.obj_classes        = obj_classes
        self.class_to_idx       = class_to_idx
        self.show_conv_filters  = True
        self.show_heatmap       = True
        
    def explain(self, im, cls):
        heatmap = Heatmap(self.model, self.obj_classes)                
        if self.show_heatmap: heatmap.explain_prediction_heatmap(im, self.obj_classes[cls])     

explainer = Explainer(model, obj_classes, class_to_idx)
#%%
def test_prediction(im=None, y=None):
    t_img = random.randint(0, len(X_train) - 1)
    if im is None: im, y = X_train[t_img], Y_train[t_img]
    pred = model.predict(np.expand_dims(im, 0))
    cls = np.argmax(y)
    explainer.explain(im, cls)    
    print("Actual: %s(%d)" % (obj_classes[cls], cls))
    for cls in list(reversed(np.argsort(pred)[0]))[:5]:
        conf = float(pred[0, cls])/pred.sum()
        print("    predicted: %010s(%d), confidence=%0.2f [%-10s]" % (obj_classes[cls], cls, conf, "*" * int(10*conf)))
    return pred

def confusion_matrix(model, dataset, X, T, accpct):
    import seaborn
    predictions = model.predict(X)
    data = np.zeros((len(obj_classes),len(obj_classes),))
    for z, c in enumerate(np.argmax(T, axis=1)): data[c][np.argmax(predictions[z])] += 1
    for z, s in enumerate(data.sum(axis=0)): data[:,z] /= (s + 1e-5)
    seaborn.set_style("whitegrid", {'axes.grid' : False})
    seaborn.heatmap(data, annot=data*100, fmt='0.0f', cmap='Wistia', xticklabels=obj_classes, yticklabels=obj_classes)
    plt.xlabel('Actual'), plt.ylabel('Predicted'), plt.title('Confusion matrix (ACC %0.2f%%)' % (accpct*100))
    plt.show(), plt.close()

def tsne_viz(model, X, Y, accpct):
    global X_embedded, xx, yy, d, predictions
    import sklearn.manifold, matplotlib.cm as cm
    predictions = model.predict(X)    
    colors = iter(cm.rainbow(np.linspace(0, 1, len(obj_classes))))
    X_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(predictions[:500])
    for d in range(len(obj_classes)):
        xx = X_embedded[Y[:500] == d, 0]
        yy = X_embedded[Y[:500] == d, 1]
        plt.scatter(xx, yy, c=next(colors), label=obj_classes[d])
        t = plt.text(np.median(xx), np.median(yy), obj_classes[d], fontsize=24)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.75})
    plt.title('T-SNE viz - Accuracy: %0.2f%%' % (accpct*100)), plt.legend()   
#%% Training code
datagen.fit(X_test)
if cfg.saveloadmodel and os.path.exists(cfg.modelid): 
    print("**** Loading existing model: %s ****" % cfg.modelid)
    model.load_weights(cfg.modelid)
#%%
vizfilt = Viz_filters(model, img_input, img_rows, img_cols)
trainstats = pd.DataFrame(columns=('Train loss', 'Test loss', 'Accuracy', 'Train accuracy'))
if os.path.exists(cfg.trainstats): trainstats = pd.read_csv(cfg.trainstats, dtype='float32')
tm = time.time()
for e in range(cfg.nb_epoch):
    print('Training... epoch=%d/%d' % (e, cfg.nb_epoch))    
    X_train, y_train, obj_classes = dataset.load_data(cfg.trainfolder, 'Train', cfg.frac, timeout=15)
    Y_train = np_utils.to_categorical(y_train, len(obj_classes))    
    loss = model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True, 
                    batch_size=cfg.batch_size), validation_data=(X_test, Y_test), steps_per_epoch=1, verbose=0)
    tloss = model.evaluate(X_train, Y_train)
    if cfg.saveloadmodel and e % 50 == 0: model.save_weights(cfg.modelid, overwrite=True)
    trainstats.loc[len(trainstats)] = (loss.history['loss'][0], loss.history['val_loss'][0], loss.history['val_acc'][0], tloss[1])
    if time.time() - tm > 60: 
        trainstats.to_csv(cfg.trainstats, index=False)
        confusion_matrix(model, dataset, X_train, Y_train, tloss[1])
        tsne_viz(model, X_train, y_train, tloss[1])        
        viz_losses(trainstats)
        t_ind = random.randint(0, len(X_train) - 1)        
        test_prediction(X_train[t_ind], Y_train[t_ind])
        tm = time.time()        
    if time.time() - tm > 600 and cfg.vizfilt_timeout > 0: vizfilt.viz_filters()
    check_memusage()
    gc.collect()
s