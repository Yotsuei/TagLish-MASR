import torchaudio
from torchaudio.transforms import Resample

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, normalize=True):
        """
        Initializes the preprocessor with a target sample rate and normalization option.
        :param sample_rate: The target sample rate for resampling audio.
        :param normalize: Whether to normalize the audio waveform.
        """
        self.sample_rate = sample_rate
        self.normalize = normalize

    def process(self, waveform, input_sample_rate):
        """
        Applies the preprocessing steps like resampling and normalization.
        :param waveform: The raw audio waveform tensor.
        :param input_sample_rate: The sample rate of the input audio.
        :return: Processed waveform.
        """
        # Resample if input sample rate is different from the desired sample rate
        if input_sample_rate != self.sample_rate:
            resampler = Resample(input_sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize the waveform to the range [-1, 1]
        if self.normalize:
            waveform = self._normalize_waveform(waveform)

        return waveform

    def _normalize_waveform(self, waveform):
        """
        Normalizes the waveform to have values between -1 and 1.
        :param waveform: The raw waveform tensor.
        :return: Normalized waveform tensor.
        """
        return waveform / waveform.abs().max()

