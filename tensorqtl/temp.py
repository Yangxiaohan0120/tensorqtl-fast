# The above-mentioned utility functions are here.  They are 'folded' (i.e.,
# hidden) by default.
# If you see the code (ie, it is not folded), in Jupyter you need to install
# codefolding by doing the following:
#
# % pip install --user jupyter_contrib_nbextensions
# % jupyter nbextension               enable codefolding/main  --user
# % jupyter nbextensions_configurator enable                   --user
#
# Note: Might need to cd to directory where Jupyter resides.
#
# Then go the Nbextensions in Jupyter and check 'Codefolding in Editor'
#
# For help, see https://github.com/ipython-contrib
# /jupyter_contrib_nbextensions/blob/master/README.md

# NOTE: Google's Colaboratory has a menu option for code folding (aka,
# code hiding).
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.optimizers import RMSprop
from datetime import datetime
import matplotlib.pyplot as p
import tensorflow as tf
import numpy as np
import multiprocessing
import IPython
import pickle
import os

# https://keras.io/datasets/#mnist-database-of-handwritten-digits
# https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles

# This Jupyter notebook uses the MNIST database of grey-scale images of the
# digits 0-9.
# Training GANs is computationally demanding - after all, we are training to
# neural networks that compute with one another.

# You might want to use the images of clothes and shoes, instead of digits 0-9.
# Just change 'useFashionMNIST = True' and all else remains the same in this
# code.
useFashionMNIST = False
# 0 = t-shirt, 1 = trouser, 2 = pullover, 3 = dress,
# 4 = cost, 5 = sandal, 6 = shirt, 7 = sneaker, 8 = bag, 9 = ankle boot

# From https://github.com/keras-team/keras/issues/4740
from tensorflow.keras import backend as K

# Some code to specify how many CPUs are used.  You might want to edit
# depending on where
# you run this code.  On the Intel DevCloud, 12 is a good setting, but '12'
# is not used
# here in case this code is run elsewhere because '12' is a high setting in
# general.
numberOfCPUsToUse = multiprocessing.cpu_count() // 2 - 0  # Your computer
# slowing down?  Set this lower!
config = tf.ConfigProto(
    intra_op_parallelism_threads=numberOfCPUsToUse,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True,
    device_count={
        'CPU': numberOfCPUsToUse
    })
session = tf.InteractiveSession(config=config)
K.set_session(session)
session.run(tf.global_variables_initializer())

# Some 'globals.'
framesForVideo = None
figureForVideo = None

# Comments below explain difference between video and movie in this program.
axForVideo = None
axForMovie = None
savedModelWeights = None
saveVideoFrames = True

# When we restart from a checkpoint file, this will record the first epoch to run.
startingEpoch = 0
startingEpoch = 0

# Display 1 in every this many sampled frames.
movieSamplingRate = 5


def getDate():
    """Produce a human-readable string containing the current time."""
    return datetime.now().strftime('%H:%M:%S %m-%d-%Y')


def chooseRandomSubset(sampleSize, items, verbosity=0):
    """Randomly select a subset of size sampleSize from these items."""
    if verbosity > 1:
        print()
        print('chooseRandomSubset: sampleSize =', sampleSize, 'items.shape = ',
              items.shape)
    return items[np.random.randint(0, items.shape[0], size=sampleSize)]


def collectDiverseFakeImages(epoch, fakeBatchSize=256, minDistance=0.0):
    """ The 'minDistance' is intended to be used if we want to require the
    generated images to be
        'sufficiently' distant from one another. Currently diversity is not
        enforced, but you might
        want to add it and experiment.
    """
    noisyInputs = np.random.uniform(
        -1.0, 1.0, size=[fakeBatchSize, randomVecSize])
    # Create some random input vectors for G, then have G generate some FAKE
    # images from them.
    fakeImages = modelG.predict(noisyInputs)

    if minDistance > 0.0:
        print("collectDiverseFakeImages: need to implement code that create a "
              "diverse of FAKE images.")

    return fakeImages, np.array(noisyInputs)


def downsizeImage(image, imageDownsize):
    """ Scale the size of this image by 'imageDownsize' - currently this code
    assumes
        images are originally 28x28 and gray scale (so one channel).
        They can only be downsized to 16x16 or 8x8
    """

    (size, size) = image.shape
    if (size != 28):
        raise ValueError(
            "downsizeImage: assumes images provided are 28x28.  Received", size,
            " x", size, "image")

    if imageDownsize != 2 and imageDownsize != 4:
        raise ValueError(
            "downsizeImage: imageDownsize must be 2 or 4.")  # Otherwise the
        # G network won't work.

    # Pad images with zeros (ie, black pixels), so when scaled by 2 or 4,
    # get size 16 or 8.
    img32 = np.zeros((32, 32))

    # Insert the image in the larger all-zero image
    img32[2:30, 2:30] = image

    # Finally, downsize the image using Python's slicing.
    return img32[::imageDownsize, ::imageDownsize]


def createRunDescription():
    """ Create a string that can be used to MARK files based on (some!)
    parameter settings."""
    if imageSize == 28:
        output = ""
    else:
        if downsize ==1:
            output_a = ""
        else:
            if downsize == 2:
                output_a = "_KmD2V2G2"
            else:
                output_a = "_KmD4V4G4"

        if use_Bias_G:
            output_b = ""
        else:
            output_b = "_noBiases"

        if useBigNetworks:
            output_c = ""
        else:
            output_c = "_small"

        if addFirstConvLayer_D:
            output_d = ""
        else:
            output_d = "_noConvD"

        if useLeakyReLUs_D:
            output_e = ""
        else:
            output_e = "_leakyReLUs"

        if useBatchNormalization_G:
            output_f = ""
        else:
            output_f = "_noBatchNorm"

        "_{}x{}"
        "_" + str(imageSize) + "x"
    return "images" + digitsToKeepString + (
        ""
        if imageSize == 28 else "_" + str(imageSize) + "x" + str(imageSize)) + (
               "" if downsize == 1 else "_KmD2V2G2" if downsize == 2 else
               "_KmD4V4G4") + ("" if use_Bias_G else "_noBiases") + (
               "" if useBigNetworks
               else "_small") + ("" if addFirstConvLayer_D else "_noConvD") + (
               "" if not useLeakyReLUs_D else "_leakyReLUs") + (
               "" if useBatchNormalization_G else "_noBatchNorm")


# Note:  the 'movie' made along the way from Numpy arrays is called the 'video'
#        and the one made from checkpointed images is called the 'movie.'


def plot_images(label,
                epoch=0,
                filename=None,
                samples=16,
                savePlotToFile=False,
                addToVideo=False):
    """ Plot a sample of REAL and FAKE images, either to the screen or to a
    file."""
    global framesForVideo, figureForVideo, axForVideo

    # Use slightly higher values for image diversity, since we look at these.
    imagesF, _ = collectDiverseFakeImages(1, samples, minImageDistance)
    imagesR = chooseRandomSubset(samples, allRealImages)

    if savePlotToFile:
        #   print('imagesF.shape =', imagesF.shape)
        #   print('imagesR.shape =', imagesR.shape)
        sqrt = int(np.sqrt(samples))
        # In subplots, first argument is ROW (y) and the second is COLUMN (
        # x), which is a bit counterintuitive.
        _, axarr = p.subplots(
            sqrt,
            2 * sqrt,
            figsize=(14, 7),
            num="REAL IMAGES on Left and FAKES on Right" + label)

        p.subplots_adjust(
            top=0.99)  # Reduce margins so as to not waste screen space.
        p.subplots_adjust(
            bottom=0.01)  # https://matplotlib.org/api/_as_gen/matplotlib
        # .pyplot.subplots_adjust.html
        p.subplots_adjust(left=0.01)
        p.subplots_adjust(right=0.99)
        p.subplots_adjust(wspace=0.02)
        p.subplots_adjust(hspace=0.02)

        for i in range(
                sqrt):  # TODO - add a whole vertical bar here as done for
            # the saved screen shots.
            for j in range(sqrt):
                axarr[i, j].imshow(imagesR[i * sqrt + j, :, :, 0], cmap='gray')
                axarr[i, j].axis('off')
                axarr[i, j + sqrt].imshow(
                    imagesF[i * sqrt + j, :, :, 0], cmap='gray')
                axarr[i, j + sqrt].axis('off')

        if filename is not None:
            print("plot_images: saving file = " + filename)
            p.savefig(filename)
        else:
            p.show()
        p.close('all')

    if addToVideo and showProgressVideoEveryThisManyEpochs >= 0:
        # https://stackoverflow.com/questions/43445103/inline-animations-in
        # -jupyter
        if framesForVideo is None:
            framesForVideo = []
        if figureForVideo is None or axForVideo is None:
            figureForVideo, axForVideo = p.subplots()
            axForVideo.axis('off')

        # Assumes array is 4x4 (actually, assumes at least 16 images
        # provided.  TODO - clean up to be more robust).
        spacer = np.ones((imageSize, 2))
        r1 = np.concatenate(
            (imagesR[0, :, :, 0], imagesR[1, :, :, 0], imagesR[2, :, :, 0],
             imagesR[3, :, :, 0], spacer, imagesF[0, :, :, 0],
             imagesF[1, :, :, 0], imagesF[2, :, :, 0], imagesF[3, :, :, 0]),
            axis=1)
        r2 = np.concatenate(
            (imagesR[4, :, :, 0], imagesR[5, :, :, 0], imagesR[6, :, :, 0],
             imagesR[7, :, :, 0], spacer, imagesF[4, :, :, 0],
             imagesF[5, :, :, 0], imagesF[6, :, :, 0], imagesF[7, :, :, 0]),
            axis=1)
        r3 = np.concatenate(
            (imagesR[8, :, :, 0], imagesR[9, :, :, 0], imagesR[10, :, :, 0],
             imagesR[11, :, :, 0], spacer, imagesF[8, :, :, 0],
             imagesF[9, :, :, 0], imagesF[10, :, :, 0], imagesF[11, :, :, 0]),
            axis=1)
        r4 = np.concatenate(
            (imagesR[12, :, :, 0], imagesR[13, :, :, 0], imagesR[14, :, :, 0],
             imagesR[15, :, :, 0], spacer, imagesF[12, :, :, 0],
             imagesF[13, :, :, 0], imagesF[14, :, :, 0], imagesF[15, :, :, 0]),
            axis=1)
        frame = np.concatenate((r1, r2, r3,
                                r4))  # r= row (wanted to fit code without
        # needed to scroll horizontally).
        framesForVideo.append(frame)

        numberOfFrames = len(framesForVideo)
        if numberOfFrames % showProgressVideoEveryThisManyEpochs == 0:
            stepSize = numberOfFrames // 10  # Sample 10 frames from the
            # start to now.
            for i in range(0, numberOfFrames + stepSize, stepSize):
                # Sample 1 in N frames from the start to now.  Added '+
                # stepSize' to make sure we get last frame.
                IPython.display.clear_output(
                    wait=True)  # Wait to clear until something ready to
                # replace it.
                j = min(i, numberOfFrames - 1)
                if framesForVideo[j] is not None:
                    axForVideo.imshow(
                        framesForVideo[j], cmap='gray', animated=True)
                    axForVideo.set_title(
                        "Left Images are REAL and Right Ones are FAKE (epoch "
                        "= " + str(j) + ")")
                    IPython.display.display(figureForVideo)


def checkpointModels(epoch):
    """ Save the state of computation to disk ('checkpointing'), so that we
    can later resume at
        the last checkpoint if we wish.
    """
    os.makedirs(
        "GANplots/savedModels/", exist_ok=True)  # Store checkpoints here.
    modelNameNoEpochNumber = createRunDescription()
    paramFileName = "GANplots/savedModels/" + modelNameNoEpochNumber + "_parameterSettings.py"
    framesFileName = "GANplots/savedModels/" + modelNameNoEpochNumber + "_frames.pickle"

    modelNameNoEpochNumber = createRunDescription()
    try:
        modelD.save(
            "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelD.h5")
        modelG.save(
            "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelG.h5")
        modelDD.save(
            "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelDD.h5")
        modelGD.save(
            "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelGD.h5")
    except NotImplementedError:
        # Not supported by Google CoLab.
        print("Model.save throws NotImplementedError, so models will not be "
              "saved.")

    with open(paramFileName, 'w') as fileName:
        fileName.write(
            str(epoch + 1))  # Add 1 since we are DONE with the provided epoch,
        # so epoch+1 is the NEXT one to do.

    if saveVideoFrames:
        with open(framesFileName, "wb") as byteFile:
            pickle.dump(framesForVideo, byteFile)
            if verbosity > 2:
                print("    Saved", str(len(framesForVideo)), "frames.")


def recoverCheckpointIfItExists():
    """ See if this run has been checkpointed.  If so, reload the saved
    neural networks
        (unless the max number of epochs has been already reached),
        set startingEpoch,
        and load the saved 'snapshots' of REAL and FAKE images generated
        after each epoch.
    """
    global framesForVideo, startingEpoch, modelD, modelG, modelDD, modelGD, startingEpoch

    modelNameNoEpochNumber = createRunDescription()
    paramFileName = "GANplots/savedModels/" + modelNameNoEpochNumber + "_parameterSettings.py"

    if os.path.exists(paramFileName) and os.path.isfile(paramFileName):
        print("The checkpoint file exists: ", modelNameNoEpochNumber)

        with open(paramFileName, 'r') as fileName:
            print("  Read the epoch number where training should resume.")
            startingEpoch = int(fileName.read())

        if (startingEpoch < maxEpochsToUse):
            print("  Load the saved neural networks.")  # Don't waste time
            # loading if they won't be used.

            # Will complain if modelD and modelG not compiled. Could simply
            # ignore, but instead they are compiled,
            # even though compilation is not otherwise needed.
            modelDname = "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelD.h5"
            modelGname = "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelG.h5"
            modelDDname = "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelDD.h5"
            modelGDname = "GANplots/savedModels/" + modelNameNoEpochNumber + "_modelGD.h5"

            if os.path.exists(modelDname): modelD = load_model(modelDname)
            if os.path.exists(modelGname): modelG = load_model(modelGname)
            if os.path.exists(modelDDname): modelDD = load_model(modelDDname)
            if os.path.exists(modelGDname): modelGD = load_model(modelGDname)

        # Always load this if it exists, so the progress movie can be watched.
        framesFileName = "GANplots/savedModels/" + modelNameNoEpochNumber + "_frames.pickle"
        if os.path.exists(framesFileName) and os.path.isfile(framesFileName):
            print("  Load the saved movie frames.")
            with open(framesFileName, "rb") as byteFile:
                framesForVideo = pickle.load(byteFile)
                if verbosity > 2:
                    print("    Read", str(len(framesForVideo)), "saved frames.")


# These next two functions are used to workaround a weakness in
# Keras/Tensorflow (as of November 2018),
# where setting 'trainable' to false and back to true, leads to a major
# slowing of the code.
# Instead we let the weights in the D network adjust while training the G
# network, but after training of G,
# we restore D's weights to the values before G was trained. This needs to be
# done after EACH batch,
# so that G doesn't rely on these changes to D's weights.
def saveModelWeights(model):  # Here weights are saved in RAM (not on the disk).
    global savedModelWeights
    savedModelWeights = model.get_weights()


def restoreModelWeights(model):
    if savedModelWeights is not None:
        model.set_weights(savedModelWeights)
    else:
        raise ValueError("restoreModelWeights: savedModelWeights=None")


# See comment above about 'video' versus 'movie.'
def showFullMovieOfSampledImages():
    """ Show a movie by sampling the saved 'snapshots' of REAL and FAKE
    images from each epoch.
        The intent is that this movie is showed at the end of training and
        the user can rewatch
        it as often as desired.
    """
    global axForMovie, movieSamplingRate

    if framesForVideo is None:
        print(
            "showFullMovieOfSampledImages(): called but framesForVideo = None")
        return
    numberOfFrames = len(framesForVideo)

    if movieSamplingRate < 1: movieSamplingRate = 1

    if axForMovie is None:
        figureForMovie, axForMovie = p.subplots()
        axForMovie.axis('off')

    userInput = "go"  # Doesn't matter as long as not one of {stop, quit, exit}.
    while (userInput != "quit" and userInput != "stop" and userInput != "exit"):
        for i in range(0, numberOfFrames + movieSamplingRate,
                       movieSamplingRate):
            # Sample 1 in N frames from the start to now.  Added '+
            # movieSamplingRate' to make sure we get last frame.
            IPython.display.clear_output(
                wait=True)  # Wait to clear until something ready to replace it.
            j = min(i, numberOfFrames - 1)
            axForMovie.imshow(framesForVideo[j], cmap='gray', animated=True)
            axForMovie.set_title(
                "Left Images are REAL and Right Ones are FAKE (epoch = " +
                str(j) + ")")
            IPython.display.display(figureForMovie)
        print("Hit ENTER in the provided text box to see the movie again (or "
              "type one of {stop, quit, exit}).")
        userInput = input()
        userInput = userInput.lower()  # Do caseless matching.

    IPython.display.clear_output()
    print("Showed a movie with numberOfFrames =", str(numberOfFrames),
          " and movieSamplingRate =", str(movieSamplingRate), "  ", getDate())
    return None  # Even with this, get two copies of the final movie frame in
    # the display.  So added the clear_output.


def trainModel(model, trainX, trainY):
    """ Train the provided model on ONE batch of examples.  Return the loss."""
    # You might want to add, say, a confusion matrix here to better
    # understand the model's errors.
    return model.train_on_batch(trainX, trainY)


###############################
#   End of utility functions  #
###############################

print("{:<30}     {}".format('Utility functions loaded.  ',
                             getDate()))  # Report when last run.
print()

# Fold to hide.  Unfold to see these functions.


def trainGAN(imageSizeToUse=28,
             digitsToKeep=True,
             downsizeToUse=1,
             useBiasesInG=False,
             numberOfEpochs=10000,
             createBigNetworks=True,
             addFirstConvLayer=True,
             alsoCreateBiggerNetworkD=True):
    init(
        imageSizeToUse=imageSizeToUse,
        digitsToKeep=digitsToKeep,
        downsizeToUse=downsizeToUse,
        useBiasesInG=useBiasesInG,
        useBigNets=createBigNetworks,
        addFirstConvLayer=addFirstConvLayer,
        alsoCreateBiggerNetworkD=alsoCreateBiggerNetworkD)

    doGANtraining(numberOfEpochs=numberOfEpochs)


def init(imageSizeToUse=28,
         digitsToKeep=True,
         downsizeToUse=1,
         useBiasesInG=False,
         addFirstConvLayer=True,
         useBigNets=True,
         alsoCreateBiggerNetworkD=True):
    """ Set up for GAN training."""

    # Note: setting of alsoCreateBiggerNetworkD NOT reflected in file names
    # produced by createRunDescription().

    global modelD, modelG, modelDD, modelGD
    global digitsToKeepString, channels, downsize, randomVecSize, verbosity
    global fakeBatchSize, imageSize, allRealImages
    global addFirstConvLayer_D, addSecondConvLayer_D, addThirdConvLayer_D, addFourthConvLayer_D, useLeakyReLUs_D
    global use_Bias_G, useBigNetworks, includeAdditionalConv2DTranspose_G, useBatchNormalization_G

    # Check for valid arguments.
    if imageSizeToUse != 28 and imageSizeToUse != 16 and imageSizeToUse != 8:
        print("init: imageSizeToUse must be one of {8, 16, 28}.")
        return
    if downsizeToUse != 1 and downsizeToUse != 2 and downsizeToUse != 4:
        print("init: downsizeToUse must be one of {1, 2, 4}.")
        return

    # We will create several Keras models.
    modelD = None  # This is the discriminator part of the GAN.
    modelG = None  # This is the generator part of the GAN.
    modelDD = None  # This holds the Discriminator so we can train it by itself.
    modelGD = None  # This holds G followed by D so we can train G, based on
    # D's predictions.

    downsize = downsizeToUse  # Varies size of D and G.  Should only be 1, 2,
    # or 4 (as checked above).
    randomVecSize = 100 // downsize  # We might want to experiment with the
    # tradeoff between
    verbosity = 1  # network size and training time until a good solution
    # obtained.
    fakeBatchSize = 128  # The Discriminator gets this many real and this
    # many fake images.
    # The Generator should get 2x this many fake images, though a parameter
    # controls the ratio.

    useBigNetworks = useBigNets  # This is also used in createRunDescription().

    # Parameters that impact D's architecture.  A number of Boolean 'flags'
    # are used to facilitate experimentation.
    addFirstConvLayer_D = addFirstConvLayer  # If false, use a network with
    # one HU layer.
    addSecondConvLayer_D = useBigNetworks and addFirstConvLayer_D
    addThirdConvLayer_D = useBigNetworks and alsoCreateBiggerNetworkD
    addFourthConvLayer_D = useBigNetworks and alsoCreateBiggerNetworkD
    useLeakyReLUs_D = False  # Otherwise simply use regular ReLU's.

    # Parameters that impact G's architecture, again to facilitate
    # experimentation.
    use_Bias_G = useBiasesInG  # If false, might reduce node collapse since G
    # cannot use biases
    #                                        to fit its training data while
    #                                        largely ignoring the random
    #                                        vector's values.
    includeAdditionalConv2DTranspose_G = useBigNetworks
    useBatchNormalization_G = True  # Based on some experiments done with
    # this code, this is necessary.

    if verbosity > 0:
        print()
        print("Read the MNIST dataset.", getDate())
    # Read in either the NIST 'digits' image dataset or the 'fashion' variant.
    if useFashionMNIST:
        (allRealImagesRaw, yAllRealImagesRaw), (
            _,
            _) = fashion_mnist.load_data()  # 60K train and 10K test examples.
    else:
        (allRealImagesRaw, yAllRealImagesRaw), (
            _, _) = mnist.load_data()  # 60K train and 10K test examples.

    if (imageSizeToUse != 28):
        imageDownsize = 2 if imageSizeToUse == 16 else 4
        allRealImagesRaw = np.array(
            list(
                map(lambda image: downsizeImage(image, imageDownsize),
                    allRealImagesRaw)))

    # Allow to focus on a SUBSET of all the digits.
    if digitsToKeep == True or digitsToKeep == "":  # See if USE ALL TEN
        # DIGITS specified.
        digitsToKeepString = "All"
    else:
        digitsToKeepString = ""
        for d in sorted(digitsToKeep):
            digitsToKeepString += str(d)

    if digitsToKeep == True or digitsToKeep == "":
        allRealImages = allRealImagesRaw
        yAllRealImages = yAllRealImagesRaw
    else:  # Collect the desired digit(s).
        allRealImages = list()
        yAllRealImages = list()
        for i in range(len(yAllRealImagesRaw)):
            if yAllRealImagesRaw[i] in digitsToKeep:
                allRealImages.append(allRealImagesRaw[i])
                yAllRealImages.append(yAllRealImagesRaw[i])
        allRealImages = np.array(allRealImages)
        yAllRealImages = np.array(yAllRealImages)

    if verbosity > 1:
        print('  digitsToKeep =', 'All' if digitsToKeep == True
        else digitsToKeep, '  number of examplesCollected =',
              len(allRealImages))

    allRealImages = allRealImages.astype(
        np.float32)  # Lets's make sure images are floats and not unsigned ints.

    if len(allRealImages.shape) == 2:
        _, two = allRealImages.shape
        sqrt = int(np.sqrt(two))
        allRealImages = allRealImages.reshape(-1, sqrt, sqrt, 1)
    elif len(allRealImages.shape) == 3:
        (_, img_rows, img_cols) = allRealImages.shape
        channels = 1
        allRealImages = allRealImages.reshape(-1, img_rows, img_cols, channels)

    (numbOfRealImages, imageSize, _, channels) = allRealImages.shape

    maxPixel = allRealImages.max()  # Scales images to be in [0,1] - some are
    # stored as unsigned ints, so in [0, 255].
    if verbosity > 3:
        minPixel = allRealImages.min()
        print('    minPixel =', minPixel, 'maxPixel = ', maxPixel)
    if maxPixel > 1: allRealImages /= maxPixel

    if verbosity > 2:
        print()
        print('    numberOfRealImages =', str(numbOfRealImages), 'imageSize =',
              str(imageSize), 'x', str(imageSize), 'channels =', str(channels),
              '   ', getDate())


###############################
#   End of setup functions  #
###############################

print("{:<30}     {}".format('Init functions loaded.  ',
                             getDate()))  # Report when last run.
print()

# Fold to hid the code.  Unfold to see it.


def createDiscriminator(verbosity=0):
    """ The Discriminator (see Figuure 1).  A standard Convolutioal Neural
    Network (CNN), though one can
      also simply use a single layer of fully connected hidden units (
      addFirstConvLayer_D = False).
    """

    dropoutIn_D = 0.10  # Drop out rate for the image (i.e., the input units).
    dropoutHU_D = 0.50  # Drop out rate for hidden units.

    numberOfHUsForOneLayerDiscrim = 256  # Used if NO convolution layers (
    # i.e., addFirstConvLayer_D = False).

    filters1_D = 64 // downsize
    strides1_D = 2  # Rather than using pooling, simply 'down sample' the
    # input image using strides of size 2.
    kernelSize1_D = 5

    filters2_D = 2 * filters1_D  # Double the number of filters but half the
    # 'image' size.
    strides2_D = 2  # So total number of hidden units is constant, though
    # that isn't necessary.
    kernelSize2_D = 5

    filters3_D = 2 * filters2_D  # Ditto.
    strides3_D = 2
    kernelSize3_D = 5

    filters4_D = 2 * filters3_D  # No halving of the 'image' this time.
    strides4_D = 1
    kernelSize4_D = 5

    modelD = Sequential()

    # Various Boolean variables control actually how much of Figure 1 is
    # constructed.
    # This allows experimentation on network complexity versus solution
    # quality as a function of training time.
    if addFirstConvLayer_D:
        if dropoutIn_D > 0.0:
            modelD.add(Dropout(dropoutIn_D))  # Drop out some image pixels.
        modelD.add(
            Conv2D(
                filters1_D,
                kernelSize1_D,
                input_shape=(imageSize, imageSize, channels),
                strides=strides1_D,
                padding='same'))  # Cannot put LeakyReLU here if
        # saving models de to Keras bug.
        modelD.add(LeakyReLU(alpha=0.1)) if useLeakyReLUs_D else modelD.add(
            Activation('relu'))
        modelD.add(Dropout(dropoutHU_D))

        if addSecondConvLayer_D:
            modelD.add(
                Conv2D(
                    filters2_D,
                    kernelSize2_D,
                    strides=strides2_D,
                    padding='same'))
            modelD.add(LeakyReLU(alpha=0.1)) if useLeakyReLUs_D else modelD.add(
                Activation('relu'))
            modelD.add(Dropout(dropoutHU_D))

            if addThirdConvLayer_D:
                modelD.add(
                    Conv2D(
                        filters3_D,
                        kernelSize3_D,
                        strides=strides3_D,
                        padding='same'))
                modelD.add(
                    LeakyReLU(alpha=0.1)) if useLeakyReLUs_D else modelD.add(
                    Activation('relu'))
                modelD.add(Dropout(dropoutHU_D))

                if addFourthConvLayer_D:
                    modelD.add(
                        Conv2D(
                            filters4_D,
                            kernelSize4_D,
                            strides=strides4_D,
                            padding='same'))
                    modelD.add(LeakyReLU(
                        alpha=0.1)) if useLeakyReLUs_D else modelD.add(
                        Activation('relu'))
                    modelD.add(Dropout(dropoutHU_D))

        modelD.add(
            Flatten())  # Create a final 'flat' layer of hidden units that
        # fully connect to the output unit/
    else:
        # Here we simply create one hidden-unit layer so we can see if how
        # much a CNN helps our GAN.
        modelD.add(Flatten())
        if dropoutIn_D > 0.0:
            modelD.add(Dropout(dropoutIn_D))  # Drop out some image pixels.
        modelD.add(
            Dense(
                numberOfHUsForOneLayerDiscrim,
                input_shape=(imageSize, imageSize, channels)))
        modelD.add(LeakyReLU(alpha=0.1)) if useLeakyReLUs_D else modelD.add(
            Activation('relu'))

    # Finally, create the single output unit.  We want its value to be in [0,
    # 1], so use a sigmoid.
    modelD.add(Dense(1, activation='sigmoid'))

    return modelD


###############################
#   End of Discriminator  #
###############################

print("{:<30}     {}".format('Discriminator function loaded.  ',
                             getDate()))  # Report when last run.
print()

# Fold to hid the code.  Unfold to see it.


def createGenerator(verbosity=0):
    """ The Generator.  See Figure 2. """

    dropoutIn_G = 0.05  # Given the input is random, less drop out needed.
    dropoutHU_G = 0.33

    filters1_G = 256 // downsize
    size1_G = imageSize // 4  # There are two 2x2 'up samplings' below and at
    # the end we want imageSize x imageSize.

    filters2_G = max(16,
                     filters1_G // 2)  # Use fewer filters as the 'image' grows.
    kernelSize2_G = 5  # You might want to experiment with the kernel size,
    # but they are a lot of parameters to vary.

    filters3_G = max(8, filters2_G // 2)
    kernelSize3_G = 5

    filters4_G = max(4, filters3_G // 2)
    kernelSize4_G = 5

    filters5_G = 1
    kernelSize5_G = 5

    modelG = Sequential()

    # Do an 'inverse flatten.'  Go from a 1D vector of random numbers to
    # size1_G x size1_G 'image'
    # across filters1_G filters that each are trained by backpropagation.
    # ('Image' is in quotes since the output of each filter in each hidden
    # layer in G is a 2D object of numbers.)
    modelG.add(
        Dense(
            size1_G * size1_G * filters1_G,
            use_bias=use_Bias_G,
            input_dim=randomVecSize))
    if useBatchNormalization_G: modelG.add(BatchNormalization(momentum=0.9))
    modelG.add(Activation('relu'))
    modelG.add(Reshape((size1_G, size1_G, filters1_G)))
    if dropoutIn_G > 0: modelG.add(Dropout(dropoutIn_G))

    # Use Conv2DTranspose to create an 'image' that is twice as large on each
    # dimension (so four times as larger overall).
    modelG.add(
        Conv2DTranspose(
            filters2_G,
            kernelSize2_G,
            strides=2,
            padding='same',
            use_bias=use_Bias_G))
    # Conv2DTranspose is sometimes called Deconvolution.
    if useBatchNormalization_G: modelG.add(BatchNormalization(momentum=0.9))
    modelG.add(Activation('relu'))
    if dropoutHU_G > 0: modelG.add(Dropout(dropoutHU_G))

    # Again use Conv2DTranspose to create an 'image' that is twice as large
    # on each dimension.
    modelG.add(
        Conv2DTranspose(
            filters3_G,
            kernelSize3_G,
            strides=2,
            padding='same',
            use_bias=use_Bias_G))
    if useBatchNormalization_G: modelG.add(BatchNormalization(momentum=0.9))
    modelG.add(Activation('relu'))
    if dropoutHU_G > 0: modelG.add(Dropout(dropoutHU_G))

    # If requested, create another Conv2DTranspose, but this time do not
    # alter the 'image' size.
    if includeAdditionalConv2DTranspose_G:
        modelG.add(
            Conv2DTranspose(
                filters4_G, kernelSize4_G, padding='same', use_bias=use_Bias_G))
        if useBatchNormalization_G: modelG.add(BatchNormalization(momentum=0.9))
        modelG.add(Activation('relu'))
        if dropoutHU_G > 0: modelG.add(Dropout(dropoutHU_G))

    # Create the final layer that is an image of the size of our REAL images.
    modelG.add(
        Conv2DTranspose(
            filters5_G, kernelSize5_G, padding='same', activation='sigmoid'))

    return modelG


###############################
#   End of Generator #
###############################

print("{:<30}     {}".format('Generator function loaded.  ',
                             getDate()))  # Report when last run.
print()

# Fold to hid the code.  Unfold to see it.


def doGANtraining(numberOfEpochs=1000):
    """ Create and 'compile' the various networks, then alternate between
    training the
        D and G networks for as many epochs as requested.  Collect samples of
        REAL and FAKE
        images along the way; save them to disk and show videos of progress
        now and then.
    """
    global modelD, modelG, modelDD, modelGD, minImageDistance

    modelD = createDiscriminator(verbosity)  # See Figure 1.
    modelG = createGenerator(verbosity)  # See Figure 2.

    modelDD = Sequential()  # Create a model just using the discriminator
    #                       (so we can set its training parameters
    #                       independently).
    modelDD.add(modelD)

    modelGD = Sequential()  # Compose the discriminator and the generator.
    modelGD.add(modelG)  # See Figure 3.
    modelGD.add(modelD)

    if verbosity > 1:  # Show the user the models created.
        modelD.summary()
        input('Check out the Discriminator model summary above, then hit '
              'RETURN.  Waiting ...')
        modelG.summary()
        input('Check out the Generator model summary above, then hit RETURN.  '
              'Waiting ...')
        modelGD.summary()
        input('Check out the Generator + Discriminator model summary above, '
              'then hit RETURN.  Waiting ...')

    # Compile the various models. ADAM uses more storage, so we won't use it.
    modelD.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0002, decay=6e-8),
        metrics=['accuracy'])  # Not necessary to compile modelD and
    # modelG, but the checkpointing
    #                                           system leads to a complaint
    #                                           about modelD and modelG not
    #                                           being compiled.
    modelG.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0002, decay=6e-8),
        metrics=['accuracy'])  # Ditto.

    modelDD.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0002, decay=6e-8),
        metrics=['accuracy'])
    modelGD.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0002, decay=6e-8),
        metrics=['accuracy'])  # b

    # You might want to use this (see collectDiverseFakeImages) to help
    # ensure that the sample
    # of fake images has some variation, to help reduce the odds of mode
    # collapse.
    minImageDistance = 0.0  # Per pixel (on average) might be one way to
    # intepret this.
    #                          Another might be to use the prediction scores
    #                          of the D network.

    os.makedirs(
        'GANplots',
        exist_ok=True)  # Store snapshots of REAL and FAKE images here.

    recoverCheckpointIfItExists()  # Old results saved to disk? If no longer
    # wanted, you need to manually delete the files!
    for epoch in range(startingEpoch, numberOfEpochs):
        if verbosity > 1:
            print()
            print("---------------")
            print("  epoch = ", epoch, "   ", getDate())
            print("---------------")

        fakeImages, randomInputs = collectDiverseFakeImages(
            epoch, fakeBatchSize)
        realImages = chooseRandomSubset(fakeBatchSize, allRealImages)
        allImages = np.concatenate((fakeImages, realImages))

        yD = np.ones([len(allImages)])  # The REAL images get y = 1.
        yD[0:len(fakeImages)] = 0  # The FAKE images get y = 0.

        if verbosity > 1:
            print()
            print("Train the Discriminator on a batch of", len(allImages),
                  "examples.")
        lossD = trainModel(modelDD, allImages, yD)
        if verbosity > 0:
            print(
                "  D  loss = {:7.4f}, accuracy = {:6.4f} ".format(
                    lossD[0], lossD[1]),
                end="")

        batchSizeMultipler = 2  # If set to 2, then G and D get the same
        # number of training examples per epoch.
        if verbosity > 1:
            print()
            print("Train the GAN (with the Discriminator's weights held "
                  "constant) on ", batchSizeMultipler * fakeBatchSize,
                  "examples.")
            # Note: see the notes by the definition of saveModelWeights()
            #       about why we have to 'fake' holding D's weights constant.

        _, randomInputs = collectDiverseFakeImages(
            epoch, batchSizeMultipler * fakeBatchSize)

        yGD = np.ones(
            [len(randomInputs)])  # These are all FAKES, but the Generators
        # 'wants' the Discriminator
        #                                    part of modelGD to predict 1 (
        #                                    i.e., be fooled).

        saveModelWeights(
            modelD)  # See the notes by the definition of saveModelWeights().
        lossGD = trainModel(modelGD, randomInputs, yGD)
        restoreModelWeights(
            modelD)  # Undo impact of changing the D part of the GD model.

        if verbosity > 0:  # Report results as training progresses.
            print("  GD loss = {:7.4f}, accuracy = {:6.4f}  {}    e{}".format(
                lossGD[0], lossGD[1], getDate(), epoch))

        # Display or save a sample of REAL and FAKE images.
        plot_images(
            ": epoch = " + str(epoch + 1) + ", batchSize = " +
            str(batchSizeMultipler * fakeBatchSize),
            filename="GANplots/plot_" + createRunDescription() + "_epoch" +
                     str(epoch + 1) + ".png",
            samples=16,
            savePlotToFile=(checkpointEveryThisManyEpochs > 0 and
                            (epoch + 1) % checkpointEveryThisManyEpochs == 0 or
                            (epoch == numberOfEpochs - 1)),
            addToVideo=showProgressVideoEveryThisManyEpochs > 0)

        # Some notes about the above function call (to plot_images).
        #  - png files are smaller than jpg by factor of 5-6
        #  - there need to be at least 16 samples for the 'addToVideo' to
        #  work properly
        #  - we want to make sure we plot and save after the last epoch,
        #    even if it doesn't match checkpointEveryThisManyEpochs

        # Every 'checkpointEveryThisManyEpochs' epochs, checkpoint the files.
        #   Note: the code that checkpoints to networks and sampled images
        #   uses createRunDescription()
        #         to label the checkpoint files (in the savedModels directory).
        #         If variables not in createRunDescription() are changed,
        #         the loaded checkpoint file will
        #         differ from the actual parameter settings and results will
        #         be corrupted.
        if (epoch + 1) % checkpointEveryThisManyEpochs == 0:
            checkpointModels(epoch)


###############################
#   End of main GAN function  #
###############################

print("{:<30}     {}".format('GAN-training function loaded.  ',
                             getDate()))  # Report when last run.
print()

# Fold to hide the code.  Unfold to see it.

# Set parameters to specify the details of GAN training.

showProgressVideoEveryThisManyEpochs = 50  # If not a positive value,
# don't show progress during training.
checkpointEveryThisManyEpochs = 100  # Save networks and sampled images to
# disk every this many epochs (if > 0).
movieSamplingRate = 100  # After training, show every Nth generated frame of
# real and fake images.

digitsToUse = True  # If True, than all 10 digits (0-9) used.
#                    Any subset (e.g., {2, 7} or {1, 3, 5, 7, 9}) other than the empty set can be used.
#                    The empty set ({}) is interpreted as 'use all 10 digits.'
digitsToUse = {2, 7}

imageSizeToUse = 28  # 16 and 8 are also valid settings.
useBiasesInG = False  # Allow biases to be learned in G?  They might make 'mode collapse' more likely.
big = True  # Determines if 'extra' components are added to the D and G networks.
downsize = 1  # Scale the nuber of filters (See Figures 1 and 2) by 1, 2, or 4.  Impacts network sizes.
maxEpochsToUse = 2500  # Remember that this later can be increased and the code re-run from the last checkpoint.

trainGAN(
    imageSizeToUse=imageSizeToUse,
    digitsToKeep=digitsToUse,
    downsizeToUse=downsize,
    useBiasesInG=useBiasesInG,
    numberOfEpochs=maxEpochsToUse,
    createBigNetworks=big,
    alsoCreateBiggerNetworkD=big)

print()
print("Done training the GAN!")
print()
showFullMovieOfSampledImages()
print()
print("Done playing 'GAN, the Movie.'")
print()

# Some notes on additional parameter settings and their defaults.
# ---------------------------------------------------------------
#
# Cannot use trainGAN() to set these, though createBigNetworks and alsoCreateBiggerNetworkD impact several of them:
#
#   addFirstConvLayer       = True # If False, then a single hidden layer used in D, and addSecondConvLayer_D ignored.
#   addSecondConvLayer_D    = True (also adds 3rd and 4th, and sets includeAdditionalConv2DTranspose_G=True)
#   useBatchNormalization_G = True  # Performance poor without this.  Feel free to try without, if curious.
#   useLeakyReLUs           = False # You might want to see if Leaky ReLU's do better in this code than standard ReLU's.
#   randomVecSize           = 100   # Size of the input to G.  Gets divided by downsize.
#   fakeBatchSize           = 128   # Number of REAL and FAKE examples given to D. Twice this is the default number G gets.
