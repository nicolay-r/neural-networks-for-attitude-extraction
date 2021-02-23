from arekit.common.experiment.neutral.annot.factory import get_annotator_type
from exp_io import CustomNetworkExperimentIO


class CustomNetworkEvaluationExperimentIO(CustomNetworkExperimentIO):

    def _get_neutral_annot_name(self):
        """ We use custom implementation as it allows to
            irresect from NeutralAnnotator instance.
        """
        scaler = self._experiment.DataIO.LabelsScaler
        annot_type = get_annotator_type(labels_count=scaler.LabelsCount)
        return annot_type.name

