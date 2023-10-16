import librosa as librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = 'model.h5'
SAMPLE_TO_CONSIDER = 22050


# Using a class provide ability to reuse the code , which makes the program more efficient
class _Keyword_Spotting_Service:

    """ Singleton class for keyword spotting inference with trained models.


    : param model: Trained model 

    """

    model = None
    _mapping = [

        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"

    ]
    _instance = None

    def predict(self, file_path):
        """
        : Param file_path (str): path to audio file to predict the
        : return predicted_keyword (str): Keyword predicted by the model    
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # We need a 4-dim array to feed to the model for prediction : (# samples , # time steps , # coefficients , 1 )
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # Get the predict label
        predictions = self.model.predict(MFCCs)
        # Get the index of the maximum value
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract MFCCs from audio file .
        : param file_path (str): path of audio file 
        : param n_fft (int): Interval we consider to apply SIFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT . Measured in # of samples

        : return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps , # coefficients)
        """

        # Load audio file

        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLE_TO_CONSIDER:
            # Ensure cosistensy of the length of the signal
            signal = signal[:SAMPLE_TO_CONSIDER]

            # Extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)

        return MFCCs.T  # Transpose because cof the flatten layer


def Keyword_Spotting_Service():
    """
    Factory function for keyword_spotting_service class.
    : return _keyword_Spotting_Service._instance (_Keyword_Spotting_Service):

    """

    # Ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(
            SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # CReate 2 instance of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    assert kss is kss1

    # Make a prediction
    keyword = kss.predict('down.wav')
    keyword1 = kss.predict('on.wav')
    print(
        f'The target is down & the predict is  {keyword}  \nThe target is on  & the predicted is {keyword1}')
