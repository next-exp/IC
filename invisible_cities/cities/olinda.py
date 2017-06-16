from __future__ import print_function
import sys

from glob import glob
from time import time
import numpy as np
import tables as tb
import os
import matplotlib.pyplot as plt
from matplotlib.patches         import Ellipse

from keras.models               import Model
from keras.models               import load_model
from keras.models               import Sequential
from keras.layers               import Input
from keras.layers               import Dense
from keras.layers               import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core          import Flatten

from   invisible_cities.core                  import fit_functions        as fitf
from   invisible_cities.core.log_config       import logger
from   invisible_cities.core.configure        import configure
from   invisible_cities.core.dnn_functions    import read_xyz_labels
from   invisible_cities.core.dnn_functions    import read_pmaps
from   invisible_cities.cities.base_cities           import KerasDNNCity
from   invisible_cities.reco.dst_io           import XYcorr_writer

from   invisible_cities.reco.corrections      import Correction

#from   .. core                  import fit_functions        as fitf
#from   .. core.log_config       import logger
#from   .. core.configure        import configure
#from   .. core.dnn_functions    import read_xyz_labels
#from   .. core.dnn_functions    import read_pmaps
#from   .  base_cities           import KerasDNNCity
#from   .. reco.dst_io           import XYcorr_writer
#
#from   .. reco.corrections      import Correction

class Olinda(KerasDNNCity):
    """
    The city of OLINDA performs the DNN analysis for point reconstruction.

    This city takes a set of input files (HDF5 files containing PMAPS, and
    MCTrack data if it is to be used for training).  It then reads the PMAPS
    and MCTrack data and performs a pre-processing step which prepares HDF5
    files containing:

        maps: [48, 48] matrices containing the SiPM responses
        coords: length-2 arrays containing true (x,y) points, if available

    This file is saved with the name specified in the configuration file
    DNN_DATAFILE.  Once this file is created, this process or creating
    "maps" and "coords" objects does not need to be run again until new data
    is to be input.

    The city can be run in 4 modes:
        - 'train': trains a DNN with the input events
        - 'retrain': same as 'train' but ensures that the DNN is trained
        from a new initialization
        - 'test': predicts (x,y) values for a given set of input events (PMAPS)
        for comparison with true values, which must also be given in the inputs
        - 'eval': predicts (x,y) values for a given set of input events (PMAPS)
        but unlike 'test' does not require that the true values are given
        (this is the mode that would be applied to detector data)

    Thus in general the city will be used as follows:
        - a large MC dataset will be input in 'train' mode, and the weights
        of the trained net will be saved
        - a subset of this MC dataset can be used in 'test' mode to verify
        that the net has been properly trained
        - real data can be input in 'eval' mode and the (x,y) predictions
        saved

    A summary of the key inputs to the configuration file:
        - FILE_IN: a list of input files
        - RUN_NUMBER: the run number
        - TEMP_DIR: a temporary directory to which network weights are written
        - MODE: the operating mode 'train', 'retrain', 'test', or 'eval'
        - WEIGHTS_FILE: the name of the file containing the weights of the
        neural network to be employed.  If this file does not exist, new files
        will be saved.
        - DNN_DATAFILE: the name of the datafile to which the pre-processed
        datasets (maps and coords) will be saved.  If this file already exists,
        the pre-processed data will be read from the file directly and the
        input files will be ignored.
        - OPT: the optimizer ('nadam' or 'sgd')
        - LRATE: the learning rate
        - DECAY: the learning rate decay rate
        - LOSS: the loss function (see Keras loss function names)
        - FILE_OUT: the name of the output file containing (x,y) predictions
        - NEVENTS: the number of events to be read

    """

    def __init__(self,
                 run_number   = 0,
                 files_in     = None,
                 file_out     = None,
                 temp_dir     = 'database/test_data',
                 weights_file = 'weights.h5',
                 dnn_datafile = 'dnn_datafile.h5',
                 nprint      = 10000,
                 lrate       = 0.01,
                 sch_decay   = 0.01,
                 loss_type   = 'mse',
                 opt         = 'nadam',
                 mode        = 'eval',
                 lifetime    = 1000):
        """
        Init the machine with the run number.
        Load the data base to access calibration and geometry.
        Sets all switches to default value.
        """
        KerasDNNCity.__init__(self,
                                   run_number      = run_number,
                                   files_in        = files_in,
                                   file_out        = file_out,
                                   temp_dir        = temp_dir,
                                   weights_file    = weights_file,
                                   dnn_datafile    = dnn_datafile,
                                   nprint          = nprint,
                                   lrate           = lrate,
                                   sch_decay       = sch_decay,
                                   loss_type       = loss_type,
                                   opt             = opt,
                                   mode            = mode)
        self.lifetime = lifetime

    def build_DNN_FC(self):
        """Builds a fully-connected neural network.
        """
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(48,48,1)))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=32,  activation='relu'))
        self.model.add(Dense(units=16,  activation='relu'))
        self.model.add(Dense(units=8,  activation='relu'))
        self.model.add(Dense(units=2,    activation='relu'))

    def build_DNN_conv2D(self):
        """Builds a 2D-convolutional neural network.
        """
        inputs = Input(shape=(48, 48, 1))
        cinputs = Conv2D(32, (4, 4), padding='same', strides=(4, 4), activation='relu', kernel_initializer='normal')(inputs)
        cinputs = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        cinputs = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_initializer='normal')(cinputs)
        cinputs = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        cinputs = Conv2D(256, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_initializer='normal')(cinputs)
        cinputs = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(cinputs)
        cinputs = BatchNormalization(epsilon=1e-05, axis=3, momentum=0.99, weights=None, beta_initializer='zero', gamma_initializer='one', gamma_regularizer=None, beta_regularizer=None)(cinputs)
        f1 = Flatten()(cinputs)
        f1 = Dense(units=1024, activation='relu', kernel_initializer='normal')(f1)
        f1 = Dropout(.6)(f1)
        coutput = Dense(units=2, activation='relu', kernel_initializer='normal')(f1)
        self.model = Model(inputs,coutput)


    def build_XY(self, nmax):
        """Builds the inputs and labels for a maximum of nmax events.

        The inputs will be placed in the vector X_in and the corresponding
        labels in the vector Y_in.  The corresponding event energies will be
        placed in the vector E_in.

        X_in has shape [Nevts, 48, 48 , 1]
        Y_in has shape [Nevts, 2]
        E_in has shape [Nevts, 1]
        """
        print(type(self.dnn_datafile))
        print(self.dnn_datafile)
        print("data file is {0}".format(os.path.isfile(self.dnn_datafile)))
        # construct the DNN data files if they do not yet exist for this run
        if(not os.path.isfile(self.dnn_datafile)):

            # read the pmaps from the input files
            maps, energies, drift_times, evt_numbers = read_pmaps(self.input_files, nmax,
                                                     self.id_to_coords,
                                                     3, 10000)

            # read the (x,y) labels from the input files if this is training
            if(self.mode == 'test' or self.mode == 'train' or self.mode == 'retrain'):

                logger.info("Reading labels...")
                labels, levt_numbers = read_xyz_labels(self.input_files,nmax,evt_numbers)

                # check to ensure the event numbers match
                for e1,e2 in zip(evt_numbers,levt_numbers):
                    if(e1 != e2):
                        logger.error("ERROR: Mismatch in event numbers e1 = {0}, e2 = {1}.".format(e1,e2))
                        exit()
                logger.info("Found {0} labels for {1} maps".format(len(levt_numbers),len(evt_numbers)))

            # add all slices to a single 2D projection and normalize
            sum_maps = np.zeros((len(maps), 48, 48))
            sum_energies = np.zeros((len(maps),1))
            for iw, (wmap,emap) in enumerate(zip(maps,energies)):
                sum_energies[iw] = [np.sum(emap)]
                sum_maps[iw,:,:] = np.sum(wmap,axis=2)
                msum = np.sum(sum_maps[iw,:,:])
                if(msum != 0):
                    sum_maps[iw,:,:] /= msum

            # save to a file
            f = tb.open_file(self.dnn_datafile, 'w')
            filters = tb.Filters(complib='blosc', complevel=9, shuffle=False)
            atom    = tb.Atom.from_dtype(sum_maps.dtype)
            tmaps   = f.create_earray(f.root, 'maps', atom, (0, 48, 48), filters=filters)
            for i in range(len(sum_maps)):
                tmaps.append([sum_maps[i]])
            atom    = tb.Atom.from_dtype(sum_energies.dtype)
            tenergies   = f.create_earray(f.root, 'energies', atom, (0, 1), filters=filters)
            for i in range(len(sum_energies)):
                tenergies.append([sum_energies[i]])
            dt_arr = np.reshape(drift_times,[len(drift_times),1])
            atom    = tb.Atom.from_dtype(dt_arr.dtype)
            ttimes   = f.create_earray(f.root, 'times', atom, (0, 1), filters=filters)
            for i in range(len(dt_arr)):
                ttimes.append([dt_arr[i]])

            if(self.mode == 'train' or self.mode == 'retrain' or self.mode == 'test'):
                atom    = tb.Atom.from_dtype(labels.dtype)
                tcoords = f.create_earray(f.root, 'coords', atom, (0, 3), filters=filters)
                for i in range(len(labels)):
                    tcoords.append([labels[i]])

            f.close()

        # otherwise, read in the data and labels from the file
        else:
            indata = tb.open_file(self.dnn_datafile, 'r')
            if(nmax > 0):
                in_maps = indata.root.maps[0:nmax]
                in_energies = indata.root.energies[0:nmax]
                in_times = indata.root.times[0:nmax]
                
                if(self.mode == 'train' or self.mode == 'retrain' or self.mode == 'test'):
                    in_coords = indata.root.coords[0:nmax]
            else:
                in_maps = indata.root.maps
                in_energies = indata.root.energies
                in_times = indata.root.times
                
                if(self.mode == 'train' or self.mode == 'retrain' or self.mode == 'test'):
                    in_coords = indata.root.coords
                    
            sum_maps = np.reshape(in_maps,(len(in_maps), 48, 48))
            sum_energies = np.array(in_energies,dtype=np.float32)
            drift_times = np.array(in_times,dtype=np.float32)
            
            if(self.mode == 'train' or self.mode == 'retrain' or self.mode == 'test'):
                labels = np.array(in_coords,dtype=np.float32)
            indata.close()

        self.X_in = np.reshape(sum_maps, (len(sum_maps), 48, 48, 1))
        self.E_in = sum_energies
        self.T_in = drift_times
        if(self.mode == 'test' or self.mode == 'train' or self.mode == 'retrain'):
            self.Y_in = labels[:,:2]/400. + 0.5
        else:
            self.Y_in = None

        logger.info("-- X_in shape is {0}".format(self.X_in.shape))
        logger.info("-- Max X is {0}".format(np.max(self.X_in)))
        logger.info("-- Min X is {0}".format(np.min(self.X_in)))

    def build_model(self):
        """Constructs or reads in the DNN model to be trained.

        The model will be build from scratch if:
            - the city is initialized in 'retrain' mode
            - the city is initialized in 'train' mode and previously trained
            weights do not already exist

        Otherwise, the weights will be read from weights_file.
        """

        weights_exist = os.path.isfile(self.weights_file)

        # build the model if no weights exist or mode is 'retrain'
        if(self.mode == 'retrain' or (self.mode == 'train' and not weights_exist)):
            self.build_DNN_FC()
            self.model.compile(loss=self.loss_type, optimizer=self.optimizer,
                               metrics=['accuracy'])
            self.model.summary()

        # otherwise read in the existing weights
        elif(weights_exist):
            logger.info("Loading model from {0}".format(self.weights_file))
            self.model = load_model(self.weights_file)
        else:
            logger.error("ERROR: invalid state in function build_model")

    def check_evt(self,evt_num,ept=None):
        """Plots the event with number evt_num

        If ept is specified, it must be a length 2 array containing the
        x and y coordinates of the reconstructed point.
        """

        logger.info("Checking event {0}".format(evt_num))
        logger.info("-- Shape of X is {0}".format(self.X_in.shape))

        # set up the figure
        fig = plt.figure();
        ax1 = fig.add_subplot(111);
        fig.set_figheight(15.0)
        fig.set_figwidth(15.0)
        ax1.axis([-250, 250, -250, 250]);

        # get the SiPM map and label
        xarr = self.X_in[evt_num]
        yarr = self.Y_in[evt_num]*400. - 200.

        # convert it to a normalized map
        probs = (xarr - np.min(xarr))
        probs /= np.max(probs)

        # draw the map
        for i in range(48):
            for j in range(48):
                r = Ellipse(xy=(i * 10 - 235, j * 10 - 235), width=2., height=2.);
                r.set_facecolor('0');
                r.set_alpha(probs[i, j]);
                ax1.add_artist(r);

        # place a large blue circle for the true EL points
        xpt = yarr[0]
        ypt = yarr[1]
        mrk = Ellipse(xy=(xpt,ypt), width=4., height=4.);
        mrk.set_facecolor('b');
        ax1.add_artist(mrk);

        # place a large red circle for reconstructed points
        if(ept != None):
            xpt = ept[0]*400. - 200.
            ypt = ept[1]*400. - 200.
            mrk = Ellipse(xy=(xpt,ypt), width=4., height=4.);
            mrk.set_facecolor('r');
            ax1.add_artist(mrk);

        plt.savefig("{0}/evt_{1}.png".format(self.temp_dir,evt_num))

    def run(self, nmax):
        """
        Run the machine
        nmax is the max number of events to run
        """

        # build the X,Y data
        self.build_XY(nmax)

        # build the Keras model for point reconstruction
        self.build_model()

        if(self.mode == 'train' or self.mode == 'retrain'):
            self.train(nbatch=40,nepochs=100)
        else:
            prediction = self.evaluate()

            X = np.zeros(len(prediction))
            Y = np.zeros(len(prediction))
            E = np.zeros(len(prediction))
            T = np.zeros(len(prediction))
            for i, (ypred,energy,tval) in enumerate(zip(prediction, self.E_in, self.T_in)):
                
                X[i] = ypred[0]*400 - 200
                Y[i] = ypred[1]*400 - 200
                E[i] = energy
                T[i] = tval
            
            # correct the energies for lifetime
            print("Times from min = {0} and max = {1}; and mean = {2}".format(np.min(T),np.max(T),np.mean(T)))
            E_corr = np.zeros(len(prediction))
            for i,(e,t) in enumerate(zip(E,T)):
                E_corr[i] = e / np.exp(-t/1000/self.lifetime)

            # apply cuts
            Z = T/1000.
            
            # radial
#            X_corr = X[X**2 + Y**2 < 1e4]
#            Y_corr = Y[X**2 + Y**2 < 1e4]
#            X = X_corr
#            Y = Y_corr
#            Z = Z[X**2 + Y**2 < 1e4]
#            E_corr = E_corr[X**2 + Y**2 < 1e4]
#            T = T[X**2 + Y**2 < 1e4]
            
            # z-range
            X = X[(Z > 0) & (Z < 500)]
            Y = Y[(Z > 0) & (Z < 500)]
            E_corr = E_corr[(Z > 0) & (Z < 500)]
            T = T[(Z > 0) & (Z < 500)]

            # energy
            X = X[(E_corr > 1000) & (E_corr < 13000)]
            Y = Y[(E_corr > 1000) & (E_corr < 13000)]
            T = T[(E_corr > 1000) & (E_corr < 13000)]
            E_corr = E_corr[(E_corr > 1000) & (E_corr < 13000)]
            print(X)
            print(Y)
            print(E_corr)

            # create a Kr table
            xs, ys, es, us = \
            fitf.profileXY(X, Y, E_corr, 30, 30, [-215.,215.], [-215.,215.])

            norm_index = xs.size//2, ys.size//2
            xycorr = Correction((xs, ys), es, us, norm_strategy="index", index=norm_index)
            nevt = np.histogram2d(X, Y, (30, 30), ([-215.,215.], [-215.,215.]))[0]

            # Dump to file
            with XYcorr_writer(self.output_file) as write:
                write(*xycorr._xs, xycorr._fs, xycorr._us, nevt)
                
            # set up the figure
            fig = plt.figure();
            #ax1 = fig.add_subplot(111);
            fig.set_figheight(15.0)
            fig.set_figwidth(15.0)
    
            plt.hist(E_corr,bins=50)
            plt.savefig("{0}/plt_energies_all.png".format(self.temp_dir))

#            if(self.mode == 'test'):
#                
#                # print true vs. predicted if in test mode
#                for ytrue,ypred in zip(self.Y_in, prediction):
#                    xt = ytrue[0]*400 - 200
#                    yt = ytrue[1]*400 - 200
#                    xp = ypred[0]*400 - 200
#                    yp = ypred[1]*400 - 200
#
#                    err = np.sqrt((xt - xp)**2 + (yt - yp)**2)
#                    logger.info("true = ({0},{1}); predicted = ({2},{3}), err = {4}".format(xt,
#                          yt,xp,yp,err))

        return len(self.X_in)


def OLINDA(argv = sys.argv):
    """OLINDA DRIVER"""
    CFP = configure(argv)

    files_in    = glob(CFP.FILE_IN)
    files_in.sort()
    print("input files = {0}".format(files_in))

    fpp = Olinda(run_number  = CFP.RUN_NUMBER,
                 files_in    = files_in,
                 temp_dir    = CFP.TEMP_DIR,
                 mode        = CFP.MODE,
                 weights_file = CFP.WEIGHTS_FILE,
                 dnn_datafile = CFP.DNN_DATAFILE,
                 opt             = CFP.OPT,
                 lrate           = CFP.LRATE,
                 sch_decay       = CFP.DECAY,
                 loss_type       = CFP.LOSS,
                 lifetime        = CFP.LIFETIME)

    fpp.set_output_file(CFP.FILE_OUT)
    fpp.set_compression(CFP.COMPRESSION)
    fpp.set_print(nprint = CFP.NPRINT)

    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))

    return nevts, nevt

if __name__ == "__main__":
    OLINDA(sys.argv)
