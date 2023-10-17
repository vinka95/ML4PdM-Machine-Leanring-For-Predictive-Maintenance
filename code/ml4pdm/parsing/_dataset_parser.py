"""Contains a DatasetParser class with methods to parse from and save to a PDMFF file
"""
import re

from ml4pdm.data import NUMERIC, SETOFLABELS, AttributeType, Dataset

# Constants
DESCRIPTION = '%'
COMMENT = '%'
RELATION = '@RELATION'
ATTRIBUTE = '@ATTRIBUTE'
DATA = '@DATA'
TARGET = '@TARGET'


class DatasetParser:
    """Contains methods to parse from a PDMFF file and save a dataset object to a PDMFF file.
    """

    def __init__(self):
        self.columns_list = []
        self.t_row = []

    def _parse_desc(self, d_row, state):
        """Parses description row from PDMFF file and returns description as a string

        :param d_row: Contains description row contents from PDMFF file.
        :type d_row: str
        :param state: Contains the current stage of parsing in PDMFF file.
        :type state: str
        :return: Returns description of the PDMFF file
        :rtype: str
        """
        desc = ' '
        if state == DESCRIPTION:
            desc = re.sub(r'^\%( )?', '', d_row)
        return desc

    def _parse_attr(self, a_row):
        """Parses attribute row from PDMFF file and returns attribute name and attribute type.

        :param a_row: Contains attribute row contents from PDMFF file.
        :type a_row: str
        :return: Returns attribute name and attribute type
        :rtype: str, subclass of AttributeType
        """
        _, attr = a_row.split(' ', 1)
        attr_name, attr_type = attr.split(' ', 1)
        self.columns_list.append(attr_name)
        return attr_name, AttributeType.parse(attr_type)

    def _parse_target(self, trow):
        """Parses target row from PDMFF file and returns target name and target type.

        :param trow: Contains target row contents from PDMFF file.
        :type trow: str
        :return: Returns target name and target type
        :rtype: str, str
        """
        _, tar = trow.split(' ', 1)
        tar_name, tar_type = tar.split(' ', 1)
        tar_type = AttributeType.parse(tar_type)
        if not isinstance(tar_type, NUMERIC) and not isinstance(tar_type, SETOFLABELS):
            raise TypeError('The target type must be either NUMERIC or SETOFLABELS!')
        return tar_name, tar_type

    def _parse_timeseries(self, ts):
        """Parses data of timeseries attributes.

        :param ts: Contains timeseries contents. Tuples of timestamp and value.
        :type ts: str
        :return: List of timeseries tuples
        :rtype: list
        """
        lst_ts = []
        lst_temp = []
        ts = ts.lstrip('(').rstrip(')')
        lst_temp = ts.split(',')
        for i in range(len(lst_temp)):
            a, b = lst_temp[i].split(':')
            lst_ts.append(tuple((float(a), float(b))))
        return lst_ts

    def _parse_instance(self, ins_row, tar_flg):
        """Parses data of individual instance.

        :param ins_row: instance row
        :type ins_row: list
        :param tar_flg: target flag
        :type tar_flg: bool
        :return: parsed instance row
        :rtype: list
        """
        for i in range(len(ins_row)):
            if ins_row[i].startswith('('):
                ins_row[i] = self._parse_timeseries(ins_row[i])
            else:
                if not ins_row[i].isalpha():
                    ins_row[i] = float(ins_row[i])
            if i == len(ins_row) - 1 and tar_flg is True:
                self.t_row.append(float(ins_row[i]))
        return ins_row

    def _parse_data(self, file, t_flag):
        """Parses data instances under DATA section in PDMFF file.

        :param file: Path and filename of the PDMFF file.
        :type file: str
        :param target_flag: Target flag is set if the PDMFF file contains Target label.
        :type target_flag: bool
        :return: Returns data row and target row
        :rtype: list, list
        """
        data_list = []
        for row in file:
            row = row.strip()
            if row.startswith(COMMENT) or not row:
                continue
            row = row.split('#')
            row = self._parse_instance(row, t_flag)
            if t_flag is True:
                data_list.append(row[:-1])
            else:
                data_list.append(row)
        return self.t_row, data_list

    def _prepare_sov_attr(self, attr, temp_lst):
        """Prepares attributes with set of values to be written in PDMFF

        :param attr: Name of the attribute
        :type attr: str
        :param temp_lst: Set of values that an attribute can take
        :type temp_lst: list
        :return: Set of values attribute in PDMFF
        :rtype: str
        """
        sov_attr = ''
        sov_attr += '{'
        for x in temp_lst:
            sov_attr += x + ', '
            if x == temp_lst[-1]:
                sov_attr += x
        sov_attr += '}'
        sov_attr += ATTRIBUTE + '\t' + attr + '\t' + sov_attr + '\n'
        return sov_attr

    def _prepare_ts(self, ts_lst):
        """Prepares timeseries attributes to be written in PDMFF

        :param ts_lst: Contains timeseries tuples
        :type ts_lst: list
        :return: Timeseries attribute in PDMFF
        :rtype str
        """
        ts_row = ''
        for tup in ts_lst:
            if tup == ts_lst[0]:
                ts_row += '('
            ts_row += str(tup[0]) + ':' + str(tup[1])
            if tup != ts_lst[-1]:
                ts_row += ','
            else:
                ts_row += ')'
        return ts_row

    def _prepare_instances(self, d_row, len_drow):
        """Prepares individual instances to be written under DATA section in PDMFF file.

        :param d_row: Contains individual data instance row
        :type d_row: list
        :param len_drow: Length of the data instance row
        :type len_drow: int
        :return: Data instance to be written in PDMFF
        :rtype: str
        """
        f_row = ''
        for j in range(len_drow):
            if isinstance(d_row[j], list):
                lst_tmp = d_row[j]
                f_row += self._prepare_ts(lst_tmp)
            else:
                f_row += str(d_row[j])
            if j != len_drow - 1:
                f_row += '#'
        return f_row

    def _prepare_data(self, data, target):
        """Prepares data to be written under DATA section in PDMFF file.

        :param data: Contains data instances to be written under DATA section of the pdmff file
        :type data: list
        :param target: Contains target labels associated to data instances
        :type target: list
        :return: Data instances to be written in PDMFF
        :rtype: str
        """
        f_data = ''
        # Encode data instances in PDMFF file
        if len(data) != 0:
            f_data += DATA + '\n'
            len_data = len(data)
            for i in range(len_data):
                f_data += '%Instance ' + str(i+1) + ' \n'
                len_data_inst = len(data[i])
                data_row = data[i]
                f_data += self._prepare_instances(data_row, len_data_inst)
                if target:
                    f_data += '#' + str(target[i])
                f_data += '\n'
        return f_data

    @staticmethod
    def parse_from_file(path: str):
        """Parses all the dataset elements like data, target, features etc from a PDMFF file.

        :param path: Path and filename of the PDMFF file.
        :type path: str
        :return: Dataset object that was created
        :rtype: Dataset
        """
        fp = ''
        description = ''
        relation = ''
        attributes = []
        target_name = ''
        target_type = []
        attribute_names = {}
        current_line = 0
        data_list = []

        parser_parse = DatasetParser()

        target_flag = False
        fp = path
        # open file
        f = open(fp, "r")
        # stores current stage of parsing
        current = DESCRIPTION
        f = iter(f)
        for row in f:
            current_line += 1
            # Continue parsing if it is an empty line
            row = row.strip(' \r\n')
            # Parse dataset description
            if row.startswith(DESCRIPTION):
                description = parser_parse._parse_desc(row, current)
            # parse relation name
            elif row.startswith(RELATION):
                current = RELATION
                a_row = row
                # print(row)
                _, b = a_row.split(' ', 1)
                relation = b
            # parse attributes list
            elif row.startswith(ATTRIBUTE):
                current = ATTRIBUTE
                attr_name, attr_type = parser_parse._parse_attr(row)
                attribute_names[attr_name] = current_line
                attributes.append((attr_name, attr_type))
            # parse target keyword
            elif row.startswith(TARGET):
                current = TARGET
                target_flag = True
                target_name, target_type = parser_parse._parse_target(row)
            # parse data keyword
            elif row.startswith(DATA):
                current = DATA
                break
            elif not row:
                continue
        if current == DATA:
            # parse data instances
            target_row, data_list = parser_parse._parse_data(f, target_flag)
        obj = Dataset()
        obj.data = data_list
        obj.target = target_row
        obj.features = attributes
        obj.description = description
        obj.relation = relation
        obj.target_name = target_name
        obj.target_type = target_type
        return obj

    @staticmethod
    def save_to_file(dataset_obj: Dataset, path: str):
        """Saves a dataset object to a PDMFF file.

        :param dataset_obj: Contains all dataset elements like data, target, features etc to be saved in a PDMFF file
        :type dataset_obj: Dataset
        :param path: Path and filename of the PDMFF file that is created.
        :type path: str
        """
        file_data = ''
        data = []
        target = []
        sov_lst = []
        parser_save = DatasetParser()
        # Encode as a description line in PDMFF file
        if dataset_obj.description != '':
            for row in dataset_obj.description.split('\n'):
                file_data += COMMENT + row
            file_data += '\n'
        # Encode as a RELATION statement in PDMFF file
        if dataset_obj.relation != '':
            file_data += RELATION + ' ' + dataset_obj.relation + '\n'
            file_data += '\n'
        # Encode as ATTRIBUTE statements in PDMFF file
        for i in range(len(dataset_obj.features)):
            j = dataset_obj.features[i]
            if isinstance(j[1], list):
                # Set of Values attribute
                sov_lst = j[1]
                sov_attr = j[0]
                file_data += parser_save._prepare_sov_attr(sov_attr, sov_lst)
            else:
                file_data += ATTRIBUTE + ' ' + j[0] + ' ' + str(j[1]) + '\n'
        file_data += '\n'
        # Encode as TARGET in PDMFF file
        if dataset_obj.target_name != '' and dataset_obj.target_type != '':
            file_data += TARGET + ' ' + dataset_obj.target_name + ' ' + str(dataset_obj.target_type) + '\n'
            file_data += '\n'
        data = dataset_obj.data
        target = dataset_obj.target
        data_section = parser_save._prepare_data(data, target)
        file_data += data_section
        fp = path
        # write to file
        f = open(fp, "w")
        f.write(file_data)
        f.close()

    @staticmethod
    def get_cmapss_data(file='fd001', test=False):
        """This method returns train and test instances of CMAPSS dataset based on test and file parameters.

        :param file: contains one of the file names(fd001, fd002, fd003 and fd004) in CMAPSS data, defaults to 'fd001'
        :type file: str, optional
        :param test: Test parameter is used to specify if both test and train instances of CMAPSS dataset, defaults to False.
        If test is set to True, the method returns two dataset objects(train and test). If test is False, then only the training dataset
        object is returned.
        :type test: bool, optional
        :return: This method returns one or two dataset objects that contains train and test data instances based on test parameter value.
        :rtype: Dataset
        """
        pdmff = '.pdmff'
        dir_path = 'ml4pdm/data/cmapss/'
        train = '_train'
        test_str = '_test'
        parser_cmapss = DatasetParser()
        if not file.isupper():
            file = file.upper()

        if test is True:
            train_path = dir_path + file + train + pdmff
            test_path = dir_path + file + test_str + pdmff
            train_obj = parser_cmapss.parse_from_file(train_path)
            test_obj = parser_cmapss.parse_from_file(test_path)
            return train_obj, test_obj
        else:
            train_path = dir_path + file + train + pdmff
            train_object = parser_cmapss.parse_from_file(train_path)
            return train_object
