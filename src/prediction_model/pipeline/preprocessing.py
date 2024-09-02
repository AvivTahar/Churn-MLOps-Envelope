import sys
import os
import math
import pickle
import csv
import pandas as pd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter
from io import StringIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prediction_model.config import config

invalid_contract_counter = Counter('invalid_contract_total', 'Total number of invalid contracts')
tenure_filled_counter = Counter('tenure_filled_total', 'Total number of filled tenures')
fill_phoneservice_counter = Counter('phoneservice_filled_total', 'Total number of filling PhoneService')


def is_json_empty(json_data):
    """Check if the JSON data is empty or only contains empty objects."""
    if not json_data:
        # If the list is empty
        return True
    
    # Check if all entries in the list are empty dictionaries
    for entry in json_data:
        if entry:  # If there's any non-empty dictionary
            return False
    
    return True

def filter_columns_leading_index(entry, index):
    keys = list(entry.keys())
    features_to_keep = keys[index:]
    for feature in config.FEATURES_AFTER_CONTRACT:
        if feature not in features_to_keep:
            features_to_keep.append(feature)
    filtered_entry = {key: entry[key] for key in features_to_keep}
    
    return filtered_entry

def filter_features(element):
    return {key: element[key] for key in config.FEATURES_AFTER_CONTRACT}

def find_leading_column_index_in_json(json_data, leading_column='customerID'):
    """Find the index of the leading column in the JSON data."""
    if not json_data:
        raise ValueError("JSON data is empty")

    first_entry = json_data[0]          # Assuming structure as in input_files, might change.
    keys = list(first_entry.keys())

    try:
        return keys.index(leading_column)
    except ValueError:
        raise ValueError(f"Leading column '{leading_column}' not found in JSON data")
    
def convert_dict_to_string(element):
    return ','.join(str(element.get(col, '')) for col in config.FEATURES)

def dict_to_csv_row(element, headers=None):

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    if headers:
        writer.writerow(element)
    else:
        writer.writerow(element)
    
    return output.getvalue().strip()

def csv_to_json(file_path):
    """Convert the CSV file to a JSON-like structure."""
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')

class FillTotalCharges(beam.DoFn):
    def __init__(self, default_value=2279.0):
        self.default_value = default_value

    def process(self, element):
        total_charges = element.get(config.FEATURE_TO_MODIFY, str(self.default_value))
        if total_charges is None or total_charges == '' or isinstance(total_charges, float) and math.isnan(total_charges):
            total_charges = str(self.default_value)

        if isinstance(total_charges, str):
            total_charges = total_charges.replace(' ', str(self.default_value))
        
        try:
            element['TotalCharges'] = float(total_charges)
        except ValueError:
            element['TotalCharges'] = self.default_value
        
        yield element

class FilterInvalidContract(beam.DoFn):
    def process(self, element):
        contract_value = element.get('Contract')
        valid_contracts = {'One year', 'Two year', 'Month-to-month'}
        
        if (contract_value not in valid_contracts):
            invalid_contract_counter.inc()
        else:
            # Yield only valid rows
            yield element

class OHEContract(beam.DoFn):
    def __init__(self):
        self.contract_mapping = {
            'One year': {'One year': 1, 'Two year': 0, 'Month-to-month': 0},
            'Two year': {'One year': 0, 'Two year': 1, 'Month-to-month': 0},
            'Month-to-month': {'One year': 0, 'Two year': 0, 'Month-to-month': 1}
            }
        
    def process(self, element):
        element.update(self.contract_mapping.get(element['Contract'], {'One year': 0, 'Two year': 0, 'Month-to-month': 0}))
        # TODO: ERROR HANDLING
        element.pop('Contract', None)
        yield element

class HandleContract(beam.DoFn):
    def process(self, element):
        if pd.isna(element['Contract']).any():
            # TODO:: ALERT!!
            raise ValueError("Null value found in 'Contract' feature")
        return [element]

class FillPhoneService(beam.DoFn):
    def process(self, element):
        # Check if the 'PhoneService' value is None or NaN and replace it with 'No'
        phone_service_value = element.get('PhoneService')
        
        if phone_service_value is None or \
           (isinstance(phone_service_value, float) and math.isnan(phone_service_value)):
            fill_phoneservice_counter.inc()
            element['PhoneService'] = 'No'
        
        yield element

class FillTenureWithMean(beam.DoFn):
    def __init__(self):
        self.tenure_mean = pd.read_csv(config.ORIG_DATA_PATH)['tenure'].mean()
        self.tenure_filled = tenure_filled_counter
    
    def process(self, element):
        tenure_value = element.get('tenure')
        if tenure_value is None or \
           (isinstance(tenure_value, float) and math.isnan(tenure_value)):
            self.tenure_filled.inc()
            element['tenure'] = float(self.tenure_mean)
        elif tenure_value == '':
            element['tenure'] = 0.0
        else:
            element['tenure'] = float(element['tenure'])
        
        yield element

class MapPhoneService(beam.DoFn):
    def process(self, element):
        element['PhoneService'] = 1 if element['PhoneService'].lower() == 'yes' else 0
        yield element

def run_pipeline_from_json(json_data, output_file, db_out_suffix):

    options = PipelineOptions()
    p = beam.Pipeline(options=options)
    
    if is_json_empty(json_data):
        print(f"Request is empty")
        raise RuntimeError(f'Request is empty')
    else:
        print(f"Request contains data.")

    # Find the index of the leading column
    customer_id_index = find_leading_column_index_in_json(json_data, leading_column='customerID')

    # Step 1: Convert JSON to a PCollection of dictionaries
    input_data = (p 
        | 'Create Input PCollection' >> beam.Create(json_data)
        | 'Filter Columns' >> beam.Map(filter_columns_leading_index, customer_id_index)
    )
    
    processed_data = (input_data
     | 'Filter Features' >> beam.Map(filter_features)
     | 'Filter Invalid Contracts' >> beam.ParDo(FilterInvalidContract())
     | 'OHE Contract' >> beam.ParDo(OHEContract())
     | 'Fill TotalCharges' >> beam.ParDo(FillTotalCharges())
     | 'Fill PhoneService' >> beam.ParDo(FillPhoneService())
     | 'Fill Tenure' >> beam.ParDo(FillTenureWithMean())
     | 'Map PhoneService' >> beam.ParDo(MapPhoneService())
     | 'Convert Dict to String' >> beam.Map(convert_dict_to_string)
    )

    header_str = ','.join(config.FEATURES)
    output_data = (
        (p | 'Create Header' >> beam.Create([header_str]), processed_data)
        | 'Reattach Header' >> beam.Flatten()
    )
    output_data | 'Write Output' >> beam.io.WriteToText(output_file, file_name_suffix=db_out_suffix, shard_name_template='')
    
    result = p.run()
    result.wait_until_finish()

    return True

if __name__ == '__main__':
    
    input_file = os.path.join(config.DATA_PATH, config.TEST_FILE_FIVE)
    json_data = csv_to_json(input_file)

    output_file = os.path.join(config.DATA_PATH, 'outputs/processed_output')
    db_out_suffix = '.csv'

    pipeline_success = run_pipeline_from_json(json_data, output_file, db_out_suffix)

    if pipeline_success:
        with open(os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME), 'rb') as f:
            rf_model = pickle.load(f)
        
        if isinstance(rf_model, RandomForestClassifier):
            print("The model was loaded successfully and is a RandomForestClassifier.")

        print("Number of trees in the forest:", rf_model.n_estimators)
        print("Features considered in the first tree:", rf_model.estimators_[0].n_features_in_)

        processed_df = pd.read_csv(output_file + db_out_suffix)
        predictions = rf_model.predict(processed_df)
        print(predictions)
        
