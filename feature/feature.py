import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem import rdMolDescriptors


class FeatureProcessor:
    def __init__(self, smiles_column='Smiles', mic_column='MIC', is_training=True):
        self.smiles_column = smiles_column
        self.mic_column = mic_column
        self.is_training = is_training

    def _remove_empty_columns(self, df):
        initial_cols = df.shape[1]
        df = df.dropna(axis=1, how='all')
        final_cols = df.shape[1]

        if initial_cols > final_cols:
            print(f"Removed {initial_cols - final_cols} empty columns")
        return df

    def _is_valid_smiles(self, smiles):
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            return False
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol is not None
        except:
            return False

    def _validate_smiles(self, df, step_name=""):
        smiles_col = self.smiles_column

        if smiles_col not in df.columns:
            raise ValueError(f"Dataframe does not contain column: {smiles_col}")

        initial_count = len(df)

        df = df[~df[smiles_col].isna()]
        df = df[df[smiles_col].astype(str).str.lower() != 'nan']
        df[smiles_col] = df[smiles_col].astype(str)
        df = df[df[smiles_col].str.strip() != ""]

        valid_mask = df[smiles_col].apply(self._is_valid_smiles)
        df = df[valid_mask]

        final_count = len(df)
        removed = initial_count - final_count
        if removed > 0:
            step_info = f" in {step_name}" if step_name else ""
            print(f"Removed {removed} invalid or unprocessable SMILES entries{step_info}")

        return df

    def calculate_molecular_weight(self, smiles):
        try:
            if not self._is_valid_smiles(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.ExactMolWt(mol) if mol else None
        except Exception as e:
            print(f"Error calculating molecular weight: {e}, SMILES: {smiles}")
            return None

    def calculate_hbd(self, smiles):
        try:
            if not self._is_valid_smiles(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles)
            return rdMolDescriptors.CalcNumHBD(mol) if mol else None
        except Exception as e:
            print(f"Error calculating HBD: {e}, SMILES: {smiles}")
            return None

    def calculate_hba(self, smiles):
        try:
            if not self._is_valid_smiles(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles)
            return rdMolDescriptors.CalcNumHBA(mol) if mol else None
        except Exception as e:
            print(f"Error calculating HBA: {e}, SMILES: {smiles}")
            return None

    def calculate_logp(self, smiles):
        try:
            if not self._is_valid_smiles(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles)
            return Descriptors.MolLogP(mol) if mol else None
        except Exception as e:
            print(f"Error calculating LogP: {e}, SMILES: {smiles}")
            return None

    def smiles_to_pubchem_fingerprint(self, smiles, radius=2, n_bits=881):
        try:
            if not self._is_valid_smiles(smiles):
                return np.zeros(n_bits, dtype=np.int8)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                return np.array(fp)
            return np.zeros(n_bits, dtype=np.int8)
        except Exception as e:
            print(f"Error generating PubChem fingerprint: {e}, SMILES: {smiles}")
            return np.zeros(n_bits, dtype=np.int8)

    def smiles_to_maccs_fingerprint(self, smiles):
        try:
            if not self._is_valid_smiles(smiles):
                return [0] * 167
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                return list(fp)
            return [0] * 167
        except Exception as e:
            print(f"Error generating MACCS fingerprint: {e}, SMILES: {smiles}")
            return [0] * 167

    def calculate_all_properties(self, df):
        df = self._validate_smiles(df, "property calculation")

        df['MolecularWeight'] = df[self.smiles_column].apply(self.calculate_molecular_weight)
        df['HydrogenBondDonors'] = df[self.smiles_column].apply(self.calculate_hbd)
        df['HydrogenBondAcceptors'] = df[self.smiles_column].apply(self.calculate_hba)
        df['LogP'] = df[self.smiles_column].apply(self.calculate_logp)

        prop_initial_count = len(df)
        df = df.dropna(subset=['MolecularWeight', 'HydrogenBondDonors',
                               'HydrogenBondAcceptors', 'LogP'])
        prop_final_count = len(df)

        if prop_initial_count > prop_final_count:
            print(f"Removed {prop_initial_count - prop_final_count} entries with property calculation errors")

        return df

    def add_pubchem_fingerprints(self, df, radius=2, n_bits=881):
        df = self._validate_smiles(df, "PubChem fingerprint calculation")

        print("Calculating PubChem fingerprints...")
        pubchem_fps = df[self.smiles_column].apply(
            self.smiles_to_pubchem_fingerprint,
            radius=radius,
            n_bits=n_bits
        )

        fingerprint_df = pd.DataFrame(pubchem_fps.tolist())
        fingerprint_df.columns = [f'PubChem_Fingerprint_{i + 1}' for i in range(n_bits)]

        result_df = pd.concat([df, fingerprint_df], axis=1)
        return result_df

    def add_maccs_fingerprints(self, df):
        df = self._validate_smiles(df, "MACCS fingerprint calculation")

        print("Calculating MACCS fingerprints...")
        maccs_fps = df[self.smiles_column].apply(self.smiles_to_maccs_fingerprint)

        maccs_df = pd.DataFrame(maccs_fps.tolist())
        maccs_df.columns = [f'MACCS_bit{i + 1}' for i in range(167)]

        result_df = pd.concat([df, maccs_df], axis=1)
        return result_df

    def process_data(self, df, add_properties=True, add_pubchem=True,
                     add_maccs=True, log_transform_mic=True):
        result_df = df.copy()

        result_df = self._remove_empty_columns(result_df)

        initial_count = len(result_df)
        result_df = self._validate_smiles(result_df, "initial validation")
        cleaned_count = len(result_df)
        if initial_count > cleaned_count:
            print(f"Total invalid SMILES entries removed in initial check: {initial_count - cleaned_count}")

        if add_properties:
            print("Calculating molecular properties...")
            result_df = self.calculate_all_properties(result_df)

        pre_fingerprint_cols = set(result_df.columns)

        if add_pubchem:
            result_df = self.add_pubchem_fingerprints(result_df)
        if add_maccs:
            result_df = self.add_maccs_fingerprints(result_df)

        final_pre_mic_count = len(result_df)
        result_df = self._validate_smiles(result_df, "final validation before processing")
        final_post_mic_count = len(result_df)
        if final_pre_mic_count > final_post_mic_count:
            print(f"Removed {final_pre_mic_count - final_post_mic_count} invalid entries in final check")

        if self.is_training and log_transform_mic and self.mic_column in result_df.columns:
            valid_mic_mask = result_df[self.mic_column] > 0
            if not valid_mic_mask.all():
                invalid_count = (~valid_mic_mask).sum()
                result_df = result_df[valid_mic_mask]
                print(f"Removed {invalid_count} entries with non-positive {self.mic_column} values before log transformation")

            result_df[self.mic_column] = np.log(result_df[self.mic_column])
            print(f"{self.mic_column} values have been log-transformed")

        fingerprint_cols = [col for col in result_df.columns if col not in pre_fingerprint_cols]
        if fingerprint_cols:
            null_mask = result_df[fingerprint_cols].isnull().any(axis=1)
            null_count = null_mask.sum()

            all_zero_mask = (result_df[fingerprint_cols] == 0).all(axis=1)
            all_zero_count = all_zero_mask.sum()

            filter_mask = ~(null_mask | all_zero_mask)
            result_df = result_df[filter_mask]

            total_removed = null_count + all_zero_count
            if total_removed > 0:
                print(f"Removed {total_removed} entries with invalid fingerprints:")
                print(f"  - {null_count} entries with null values in fingerprints")
                print(f"  - {all_zero_count} entries with all-zero fingerprints")

        if self.is_training and self.smiles_column in result_df.columns:
            result_df = result_df.drop(columns=[self.smiles_column])
            print(f"Removed {self.smiles_column} column after processing")

        total_removed = initial_count - len(result_df)
        print(f"Total entries removed during all processing steps: {total_removed}")

        return result_df

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def save_data(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f'Results saved to {file_path}')


def train_features(input_file, output_file, smiles_column='Smiles', mic_column='MIC'):
    processor = FeatureProcessor(smiles_column=smiles_column, mic_column=mic_column, is_training=True)
    df = processor.load_data(input_file)
    print(f"Original data rows: {len(df)}, columns: {len(df.columns)}")

    result_df = processor.process_data(df)
    processor.save_data(result_df, output_file)

    print(f"Final data rows: {len(result_df)}")
    print(f"Final data columns: {len(result_df.columns)}")
    return result_df


def file_train_features(input_file, output_file, smiles_column='Smiles', mic_column='MIC'):
    processor = FeatureProcessor(smiles_column=smiles_column, mic_column=mic_column, is_training=True)
    df = processor.load_data(input_file)
    print(f"Original data rows: {len(df)}, columns: {len(df.columns)}")

    result_df = processor.process_data(df)

    if not output_file.endswith('.csv'):
        output_file = os.path.join(output_file, 'processed_train_features.csv')

    processor.save_data(result_df, output_file)
    print(f"Final data rows: {len(result_df)}")
    print(f"Final data columns: {len(result_df.columns)}")
    return output_file


def prediction_features(input_file, output_file, smiles_column='Smiles', mic_column='MIC'):
    processor = FeatureProcessor(smiles_column=smiles_column, mic_column=mic_column, is_training=False)
    df = processor.load_data(input_file)
    print(f"Original data rows: {len(df)}")

    result_df = processor.process_data(df)
    processor.save_data(result_df, output_file)

    print(f"Final data rows: {len(result_df)}")
    print(f"Final data columns: {len(result_df.columns)}")
    return result_df

def file_prediction_features(input_file, output_file, smiles_column='Smiles', mic_column='MIC'):
    processor = FeatureProcessor(smiles_column=smiles_column, mic_column=mic_column, is_training=False)

    df = processor.load_data(input_file)
    print(f"Original data rows: {len(df)}")

    result_df = processor.process_data(df)

    if not output_file.endswith('.csv'):
        output_file = os.path.join(output_file, 'processed_prediction_features.csv')

    processor.save_data(result_df, output_file)
    print(f"Final data rows: {len(result_df)}")
    print(f"Final data columns: {len(result_df.columns)}")

    return output_file