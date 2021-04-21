'''
Extract features for training & crate train-test split
'''
# %%
import json


class EDH():
    '''
    Convenience class for operations with the EDH dataset.
    '''
    def __init__(self) -> None:
        # where is the input EDH dataset
        self.path_raw_dataset = '/home/jan/epigraphic_roads/data/EDH_text_cleaned_2021-01-21.json'
        # fields of interest
        self.keys_to_extract = [
            'type_of_inscription_clean', 'type_of_inscription_certainty',
            'height_cm', 'width_cm', 'depth_cm',
            'material_clean', 
            'type_of_monument_clean', 'type_of_monument_certainty',
            'province_label_clean', 'province_label_certainty',
            'country_clean', 'country_certainty',
            'findspot_ancient_clean', 'findspot_ancient_certainty',
            'modern_region_clean', 'modern_region_certainty',
            'findspot_modern_clean', 'findspot_modern_certainty',
            'findspot_clean', 'findspot_certainty',
            'origdate_text_clean',
            'clean_text_conservative',
            'clean_text_interpretive_word',
            'clean_text_interpretive_sentence'
        ]

    # IMPORT EXPORT
    def open_dataset(self):
        '''Load json file
        '''
        with open(self.path_raw_dataset) as fin:
            self.data_raw = json.load(fin)
    
    def save_dataset(self):
        pass


    # DATA WRANGLING
    @staticmethod
    def extract_keys(dictionary, keys):
        '''Keep only desired keys in a dictionary

        Parameters
        ----------
        dictionary : dict
            dict to filter
        keys : list
            names of keys to keep

        Returns
        -------
        dict
            curbed dict
        '''
        return {k:v for (k, v) in dictionary.items() if k in keys}


    def select_keys_dataset(self):
        '''Select fields from the data_raw

        Parameters
        ----------
        self.data_raw : list of dict
            Unprocessed EDH dataset
        
        self.keys_to_extract : list of str
            Names of fields to keep
        
        Returns
        -------
        self.data_filtered : list of dict
            EDH dataset with only specified keys
        '''
        data_filtered = [
            self.extract_keys(inscription, self.keys_to_extract) 
            for inscription in self.data_raw
            ]

        self.data_filtered = data_filtered


def main():
    # init 
    edh = EDH()
    # run through operations
    edh.open_dataset()
    edh.select_keys_dataset()
    
    return edh
# %%

if __name__ == "__main__":
    import pandas as pd
    edh = main()
    df_all = pd.DataFrame(edh.data_filtered)
    df = df_all.query('type_of_inscription_certainty == "Certain"')
    df.to_csv(
        f'/home/jan/ancient-classifier/data/210416_certain_Y/edh_{len(df)}.csv')
