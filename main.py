import argparse
from midi_parser.parse_midi import MIDI_Converter as MC
import save_to_file as stf
import keras
import tensorflow
from data_processing.preprocessing2 import Preprocessor
from neural_network.NeuralNetwork import NeuralNetwork
import logging
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Activation
import Plotter as plot
import theano



# which information to write to the file
logLevelFile = logging.DEBUG

num_cores = 8
num_GPU = 1
num_CPU = 1


# training settings
epochs = 2
batch_size = 64
render_device = "cpu"

from tensorflow.python.client import device_lib

#theano.config.device =  'gpu0'
theano.config.floatX = 'float64'

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--jsonfiles", required=False, help="Folder that holds all the jsonl input data", default="./data/midi-json/MegaMan/")
    parser.add_argument("-lw", "--loadweights", required=False, help="Path to .hdf5 file that contains weights to load in")
    parser.add_argument("-ct", "--continue_training", required=False, help="Continue training model based on loaded weights", action='store_true')
    parser.add_argument("-pn", "--predict_notes", required=False, help="Number of notes to predict", type=int, default=0)
    parser.add_argument("-lf", "--logfile", required=False, help="Set the path and name of the log file", default="./output/logging/netlog.log")
    parser.add_argument("-v", "--verbose", required=False, help="Verbose output", action='store_true')

    args = parser.parse_args()
    log = str(args.logfile)

    # enable verbose output
    logLevel = logging.DEBUG
    if args.verbose:
        print("Verbose terminal output enabled.")
    else:
        print("Verbose terminal output disabled.")
        logLevel = logging.INFO


    # get passed weights path
    weightPath = args.loadweights
    if not weightPath is None and weightPath != "":
        print("Loading weights from: " + weightPath)
    else:
        weightPath = None


    # check if paths exist
    stf.checkPath(log)
    
    stf.convertMultipleFiles("./data/midi/MegaMan/" , "./data/midi-json/")


    ###### logging configuration ######

    # create formatter
    logFormat = '%(asctime)s - [%(levelname)s]: %(message)s'
    logDateFormat = '%m/%d/%Y %I:%M:%S %p'

    formatter = logging.Formatter(fmt=logFormat, datefmt=logDateFormat)

    # create console handler (to log to console as well)
    ch = logging.StreamHandler()
    ch.setLevel(logLevel)
    ch.setFormatter(formatter)

    # configure logging, level=DEBUG => log everything
    logging.basicConfig(filename=log, level=logLevelFile, format=logFormat, datefmt=logDateFormat)

    # get the logger
    logger = logging.getLogger('musicnetlogger')
    logger.addHandler(ch)

    ###### logging configuration ######


    # print setting information
    logger.debug('Logger started.')


    # check if notes to predict value is valid
    notes_to_predict = args.predict_notes
    if notes_to_predict < 0:
        logger.error("Number of notes can not be negative!")
        return


    # get preprocessor
    preprocessor = performPreprocessing(logger, args)

    # get the network
    network = createNetworkLayout(logger, preprocessor)

    net_fit = False
    if not weightPath is None:
        if network.load_weights(weightPath):
            logger.info("Weights loaded.")
            if args.continue_training:
                network = fitNetwork(logger, network, preprocessor)
            net_fit = True
        else:
            logger.error("Failed to load weights from file: " + weightPath)
    else:
        network = fitNetwork(logger, network, preprocessor)
        net_fit = True
    plotter = plot.Plotter(network)
    plotter.plotEpochLoss("results.png", "Epoch/Loss Result")
    if net_fit and notes_to_predict > 0:

        # plot the model
        #network.plotModel("./data/plotted.png")

        # predicting
        predicted_notes = predictNotes(logger, preprocessor, network, notes_to_predict)
        print(predicted_notes)
        
    #Predicted Notes to Midi
    midi = MC.dataToMidi(predicted_notes,"predicted.mid")


def fitNetwork(logger, network, preprocessor):
    if render_device == "gpu":
        logger.info("GPU activated..")
        print("GPU activated..")
        #print(K.tensorflow_backend._get_available_gpus())

        #network.setRenderDevice(render_device)
    logger.info("Fitting model...")
    network.fit(_x=preprocessor.getNetworkData()["input"], _y=preprocessor.getNetworkData()['output'], _epochs=epochs, _batch_size=batch_size)
    return network


def performPreprocessing(logger, args):
    '''
    Performs preprocessing and returns the preprocessor.
    '''
    print("ARGS: "+args.jsonfiles)
    preprocessor = Preprocessor(logger)
    preprocessor.concatFiles(args.jsonfiles)
    logger.debug("Got dataset of length: {}".format(len(preprocessor.getDataset())))

    preprocessor.labelEncode()
    inv = preprocessor.labelEncode(True, preprocessor.getDataset())
    normds = preprocessor.normalizeDataset()

    # how many notes to predict a new note
    preprocessor.setSequenceLength(100)

    logger.info("Classes:\n{}".format(preprocessor.getLabelEncoder().classes_))
    logger.info("Inv:\n{}".format(inv[:100]))
    logger.info("Dataset:\n{}".format(preprocessor.getDataset()[:100]))
    logger.info("Dataset-Normalized:\n{}".format(normds[:100]))
    logger.info("Network Data:\n{}".format(preprocessor.createNetworkData()))

    return preprocessor


def createNetworkLayout(logger, preprocessor):
    '''
    Returns the network with the specified layout.
    '''

    # Create Neural Network
    network = NeuralNetwork()
    network.createSequentialModel()
    
    input_shape = (preprocessor.getNetworkData()['input'].shape[1], preprocessor.getNetworkData()['input'].shape[2])
    vokab_length = len(preprocessor.getLabelEncoder().classes_)

    # Add Layers

    # units = how many nodes a layer should have
    # input_shape = shape of the data it will be training
    network.add(LSTM(units=256, input_shape=input_shape, return_sequences=True))

    # rate = fraction of input units that should be dropped during training
    network.add(Dropout(rate=0.3))

    network.add(LSTM(units=512, return_sequences=True))
    network.add(Dropout(rate=0.3))

    network.add(LSTM(units=256))
    network.add(Dense(units=256))
    network.add(Dropout(rate=0.3))

    # units of last layer should have same amount of nodes as the number of different outputs that our system has
    # -> assures that the output of the network will map to our classes
    network.add(Dense(units=vokab_length))
    network.add(Activation('softmax'))

    logger.info("Compiling model...")
    network.compile(_loss='categorical_crossentropy', _optimizer='rmsprop', _metrics=['acc'])

    logger.info("Finished compiling.")
    logger.info("Model Layers: \n[]".format(network._model.summary()))

    return network


def predictNotes(logger, preprocessor, network, n_notes):

    vokab_length = len(preprocessor.getLabelEncoder().classes_)
    network_input = preprocessor.getNetworkData()['input']
    start = np.random.randint(0, len(network_input) - 1)

    # as many notes as the used sequence length
    pattern = network_input[start]
    output = []

    logger.info("Predicting {} notes...".format(n_notes))

    for i in range(n_notes):

        # reshape to row-vector
        p_input = np.reshape(pattern, (1, len(pattern), 1))

        # normalize input
        p_input = p_input / vokab_length

        # make a prediction
        # (array of predictions for all available classes of label encoding)
        prediction = network._model.predict(p_input, verbose=0)

        # get class index in label encoding / note with the highest probability
        note_index = np.argmax(prediction)
        output.append(note_index)

        # add index to pattern and remove the first entry
        pattern = np.append(pattern, note_index)
        pattern = pattern[1:]

    logger.warning("OUTPUT:\n{}".format(output))

    # get the according notes
    output = preprocessor.labelEncode(invert=True, invert_data=output)

    return output


if __name__ == '__main__':
    main()
