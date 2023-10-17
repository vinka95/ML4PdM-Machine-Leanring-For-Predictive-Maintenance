import jsonpickle
from sklearn.pipeline import Pipeline


class PipelineConfigParser:
    """Contains method to transform a sklearn pipeline into JSON and back.
    """

    @staticmethod
    def save_to_file(pipeline: Pipeline, path: str):
        """Saves a configured pipeline to a JSON file.

        :param pipeline: Pipeline that will be saved
        :type pipeline: sklearn.pipeline.Pipeline
        :param path: Path and filename of the JSON file that is created
        :type path: str
        """
        output = jsonpickle.encode(pipeline, indent=4)
        with open(path, "w") as file:
            file.write(output)

    @staticmethod
    def parse_from_string(string: str):
        """Parses a pipeline from a JSON string.

        :param string: String that contains a pipeline description in JSON format
        :type string: str
        :return: Pipeline that was decoded from the passed string
        :rtype: sklearn.pipeline.Pipeline
        """
        return jsonpickle.decode(string)

    @staticmethod
    def parse_from_file(path: str):
        """Parses a pipeline from a JSON file.

        :param path: Path and filename of the JSON file that is read
        :type path: str
        :return: Pipeline that was decoded from the file contents
        :rtype: sklearn.pipeline.Pipeline
        """
        with open(path, "r") as file:
            return PipelineConfigParser.parse_from_string(file.read())
