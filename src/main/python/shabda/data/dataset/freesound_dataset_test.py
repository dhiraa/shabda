import tensorflow as tf
from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset


class FreeSoundDatasetTest(tf.test.TestCase):

    def test_init(self):
        dataset = FreeSoundAudioDataset()
        dataset.init()
        assert (dataset.train_df.shape[0] != 0)
        assert (dataset.val_df.shape[0] != 0)
        assert (dataset.test_df.shape[0] != 0)

    def test_get_num_samples(self):
        dataset = FreeSoundAudioDataset()
        dataset.init()
        assert (dataset.train_df.shape[0] != 0)

    def test_get_lables(self):
        dataset = FreeSoundAudioDataset()
        dataset.init()
        labels = dataset.get_labels()
        assert len(labels) == 41


if __name__ == "__main__":
    tf.test.main()
