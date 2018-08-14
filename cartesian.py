import pandas as pd
from pandas import read_csv
import yaml
from functools import reduce

from os import listdir, walk, replace
from os.path import isfile, join, normpath, basename
from pandas import DataFrame
from re import sub
from unidecode import unidecode


class Cartesian:
    """Searches for folders with headers in the specified path.

    For each folder found, enters into it, and reads the header files in
    CSV format to later produce all posible combinations of them.
    """

    cfg = dict()

    def __init__(self, path):
        self.read_config(path)

    def output_name(self):
        """Returns the output filename to be generated for a given folder"""
        uc_name = basename(normpath(self.cfg['path']))
        output_name = join(self.cfg['path'],
                           'utterances_{}.csv'.format(uc_name))
        return output_name

    def rename_file(self, filename):
        """Renames an existing file adding extension '.1' or '.2' ...
        until a name is valid"""
        suffix = 1
        filename_changed = False
        while filename_changed is False:
            try:
                newfilename = '{}.{:d}'.format(filename, suffix)
                if isfile(newfilename):
                    raise(OSError)
                replace(filename, newfilename)
                filename_changed = True
                print('  File {} renamed to {}'.format(
                    basename(filename), basename(newfilename)))
            except OSError:
                suffix += 1

    def read_config(self, path):
        """Look for a config file in the folder """
        config_file = join(path, 'config.yml')
        if isfile(config_file):
            with open(config_file, 'r') as ymlfile:
                self.cfg = yaml.load(ymlfile)
                self.cfg['path'] = path
        else:
            print('DEFAULT CONFIG')
            self.cfg = {
                'tag_header': 'tag',
                'utt_header': 'utterance',
                'amr_header': 'amr',
                'com_header': 'combination_id',
                'sep': ';',
                'encoding': 'utf8',
                'max_records': 5000,
                'max_rows': 50,
                'output_dirname': 'output',
                'path': path
            }
        output_name = self.output_name()
        if isfile(output_name) is True:
            self.rename_file(output_name)

    def clean(self, text_string):
        """ Clean up a text, removing accents and strange characters,
        returning it in lower case."""
        # Quitar acentos:
        text_string = unidecode(text_string)
        # Quitar caracteres extraños
        text_string = sub('[^a-zA-Z0-9ñ:<> ]', '', text_string)
        # Dejar sólo un espacio entre palabras
        text_string = " ".join(text_string.split())
        # Poner todo en minuscula y sin espacios al principio y al final
        return text_string.strip().lower()

    def cartesian_product(self, df1, df2):
        """ Given two data frames, combine all rows from the first
        with all rows from the second one.
        """
        pd.options.mode.chained_assignment = None  # default='warn'
        df1['key'] = df2['key'] = 0
        return df1.merge(df2, on='key', how='outer').drop('key', 1)

    def cross_tables(self, path):
        """ Combine all data frames (CSV) files in a folder path into a single
        data frame with the cartesian product of all its headers.
        """
        _sep = self.cfg['sep']
        _enc = self.cfg['encoding']
        table_files = [f for f in listdir(path) if isfile(join(path, f))]
        table_files.sort()
        table = []
        for file in table_files:
            filepath = join(path, file)
            if isfile(filepath) and filepath.endswith('.csv'):
                contents = read_csv(filepath, sep=_sep, encoding=_enc)
                num_nans = contents.isnull().values.sum()
                if num_nans > 0:
                    contents = contents.dropna()
                table.append(contents[:self.cfg['max_rows']])
        return reduce(lambda df1, df2: self.cartesian_product(df1, df2), table)

    # Join the cols specified in a single string.
    def str_join(self, df, sep, cols):
        return reduce(lambda x, y: x.str.cat(y, sep=sep),
                      [df[col] for col in cols])

    def expand_amr(self, tag):
        """Given a string with a sequence of amr codes, expand the characters
        with the actual values from the dictionary passed.
        """
        a_colname = self.cfg['amr']
        amr = dict([(a_colname.get(i, ''), 1) for i in tag.split()]).keys()
        return list(filter(None, list(amr)))

    def combine_entities(self, folder):
        """ Produce all combinations of headers (entities) in a folder (path),
        producing a two-columns data frame.
        """
        path = join(self.cfg['path'], folder)
        u_tables = self.cross_tables(path)
        col_names = list(u_tables)
        uttr_headers = [col_names[i] for i in range(0, u_tables.shape[1], 2)]
        tags_headers = [col_names[i] for i in range(1, u_tables.shape[1], 2)]
        result = DataFrame({self.cfg['utt_header']: self.str_join(
                            u_tables, ' ', uttr_headers),
                            self.cfg['tag_header']: self.str_join(
                            u_tables, ' ', tags_headers),
                            self.cfg['com_header']: folder})
        return result[:self.cfg['max_records']]

    def enrich_utterances(self, utterances):
        """ Clean up text retrieved from extrange characters """
        u_colname = self.cfg['utt_header']
        utterances[self.cfg['utt_header']] = utterances[u_colname].apply(
            lambda u: self.clean(u))
        t_colname = self.cfg['tag_header']
        utterances[self.cfg['tag_header']] = utterances[t_colname].apply(
            lambda t: self.clean(t))
        # Add the AMR column with the expansion of the amr codes.
        a_colname = self.cfg['amr_header']
        utterances[a_colname] = utterances[t_colname].apply(self.expand_amr)
        # Reorder everything
        utterances = utterances[[self.cfg['utt_header'],
                                 self.cfg['tag_header'],
                                 self.cfg['amr_header'],
                                 self.cfg['com_header']]]
        # Reset index in data frame and remove index column
        utterances = utterances.reset_index().drop(['index'], axis=1)
        return utterances

    def to_csv(self, utterances):
        """Saves the utternaces to the folder as tags and headers"""
        output_name = self.output_name()
        utterances.to_csv(output_name, mode='a+', encoding='utf8')
        print('    Appended >> {}'.format(basename(output_name)))

    def product(self, save=True):
        """Generates a cartersian product of all the headers in folder"""
        all_utterances = DataFrame({self.cfg['utt_header']: [],
                                    self.cfg['tag_header']: []})
        folders = next(walk(self.cfg['path']))[1]
        for folder in folders:
            print('  Adding utts for: {}'.format(folder))
            utterances = self.combine_entities(folder)
            utterances[self.cfg['com_header']] = folder
            utterances = self.enrich_utterances(utterances)
            if save is True:
                self.to_csv(utterances)
            all_utterances = pd.concat([all_utterances, utterances],
                                       axis=0, sort=True)
        return all_utterances

    def old_product(self, save=True):
        """Generates a cartersian product of all the headers in folder"""
        folders = next(walk(self.cfg['path']))[1]
        all_utterances = DataFrame({self.cfg['utt_header']: [],
                                    self.cfg['tag_header']: []})
        # merge the combinations obtained on every folder.
        for folder in folders:
            print('  Entering: {}'.format(folder))
            utterances = self.combine_entities(folder)
            utterances[self.cfg['com_header']] = folder
            all_utterances = pd.concat([all_utterances, utterances],
                                       axis=0, sort=True)
        utterances = self.enrich_utterances(all_utterances)
        if save is True:
            self.to_csv(utterances)
        return utterances
