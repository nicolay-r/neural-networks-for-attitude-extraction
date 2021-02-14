import arekit
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.sample import InputSampleReader


def calculate_samples_count(input_samples_filepath, uint_label):
    """ Provides amount of samples which marked with the related uint_label.
        This method utilized in statistics.
    """

    samples_reader = InputSampleReader.from_tsv(
        filepath=input_samples_filepath,
        row_ids_provider=MultipleIDProvider())

    ids = set()

    total_labels = 0

    for index, row in samples_reader._df.iterrows():

        r_id = row[arekit.common.experiment.const.ID]
        r_label = row[arekit.common.experiment.const.LABEL]

        if r_id in ids:
            continue

        ids.add(r_id)
        if int(r_label) != uint_label:
            continue

        total_labels += 1

    return total_labels
