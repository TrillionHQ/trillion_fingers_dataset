"""dataset dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dataset dataset."""

  VERSION = tfds.core.Version('1.0.2')
  RELEASE_NOTES = {
      '1.0.2': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3)),
            'mask': tfds.features.Image(shape=(None, None, 3), encoding_format='png'),
            'trimap': tfds.features.Image(shape=(None, None, 1), encoding_format='png')
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'mask', 'trimap'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    print("start downloading")
    # path = dl_manager.download_and_extract('https://my-telegram-bot-dev-serverlessdeploymentbucket-ega9f2v7d5tr.s3.amazonaws.com/dataset_2.zip')
    # path = dl_manager.download_and_extract('https://drive.google.com/uc?export=download&id=1W7SzE12OwTKQ5pVYC4tSktst3fGlp6bf') #one
    path = dl_manager.download_and_extract(
        'https://drive.google.com/uc?export=download&id=1MsgJJffG0IZmuyB9xUCVRH8GWC3-nW5w') #mixed


    print("downloaded")

    # TODO(dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_images', path / 'train_masks', path / 'train_trimap'),
        'test': self._generate_examples(path / 'test_images', path / 'test_masks', path / 'test_trimap'),
        'full': self._generate_examples(path / 'full_images', path / 'full_masks', path / 'full_trimap')
    }

  def _generate_examples(self, images_path, masks_path, trimap_path):
    """Yields examples."""
    # TODO(dataset): Yields (key, example) tuples from the dataset
    for img_path in images_path.glob('*.jpg'):
      mask_name = str(img_path.name.split('.')[0]) + '.png'
      yield img_path.name, {
          'image': img_path,
          'mask': masks_path / mask_name,
          'trimap': trimap_path / mask_name
      }
