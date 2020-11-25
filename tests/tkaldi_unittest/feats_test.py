"""Test """

import torch
import tkaldi
from parameterized import parameterized

from tkaldi_unittest import utils


class PitchTest(utils.case.TestCase):
    @parameterized.expand([
        ({'sample_frequency': 8000}, ),
        ({'sample_frequency': 16000}, ),
        ({'sample_frequency': 441000}, ),
    ])
    def test_comput_kaldi_pitch(self, args):
        """compute_kaldi_pitch matches compute-kaldi-pitch-feats
        """
        frequency = 300
        sample_rate = args['sample_frequency']

        original = utils.data.get_sinusoid(
            sample_rate=sample_rate, frequency=frequency,
            num_channels=1, dtype='int16')[0]

        wave = original.to(dtype=torch.float)
        found = tkaldi.feats.compute_kaldi_pitch(wave, **args)

        path = self.get_temp_path('test.wav')
        utils.io.save_wav(path, original, sample_rate)

        _args = utils.kaldi.convert_args(**args)
        command = ['compute-kaldi-pitch-feats'] + _args + ['scp:-', 'ark:-']
        expected = utils.kaldi.run_command_scp(command, path)

        self.assertEqual(expected, found)
