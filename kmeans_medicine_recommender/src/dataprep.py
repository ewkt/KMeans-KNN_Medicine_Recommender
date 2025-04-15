import re
import pandas as pd


class DataPrep:
    def __init__(self, path):
        """
        This class is used to prepare the data for analysis.
        It includes functions to process rows, set uncommon values to 'other',
        and a main function to run the script.
        """
        self.re1 = r"shortness(\s*of\s*brea(th|t)?)?"
        self.re2 = r'[a-z_-]*covid(-19)?_exposure[a-z_-]*'
        self.path = path

    def process_row(self, row):
        """
        This function standradises the values, removing duplicates
        """
        row = row.lower().split(',')
        row = [re.sub(self.re1, 'shortness_of_breath', val) for val in row]
        row = [val.replace(' ', '_') for val in row]
        row = [re.sub(r'^e_', '', val) for val in row]
        row = [val.lstrip('_') for val in row]
        row = [val.rstrip('_') for val in row]
        row = [val.rstrip('_o') for val in row]
        row = [val.replace('on___', '') for val in row]
        row = [re.sub(self.re2, 'covid_exposure', val) for val in row]
        row = [re.sub(r'rheumatoid_arthrit(i|is)?', 'rheumatoid_arthritis', val) for val in row]
        row = [val.replace('chronic_fatiguesyndrome', 'chronic_fatigue_syndrome') for val in row]
        return row

    def set_uncommon_to_other(series):
        """
        This function takes a pandas Series and replaces uncommon values with 'other'.
        A value is considered uncommon if it appears less than 6 times in the Series.
        """
        value_counts = series.explode().value_counts()
        uncommon_values = value_counts[value_counts < 6].index.tolist()
        series = series.apply(lambda x: ['other' if val in uncommon_values else val for val in x])
        return series

    def main(self):
        """
        Main function to run the script.
        """
        #Data Imported from Kaggle:
        #https://www.kaggle.com/datasets/joymarhew/medical-reccomadation-dataset
        df_raw = pd.read_csv(self.path)

        df = df_raw.copy()
        df.drop(columns=['Name'], inplace=True)
        df.dropna(thresh=4, inplace=True) # On supprime les colonnes qui n'ont pas assez d'info

        ######## 1. AGE
        df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce', dayfirst=True)
        df['Age'] = (2025 - df['DateOfBirth'].dt.year)
        mean_age = df['Age'].mean() # On complete les ages NA avec la moyenne
        df['Age'] = df['Age'].fillna(mean_age).astype(int)

        df['is_young'] = [1 if age < 25 else 0 for age in df['Age']] # On crée des features one-hot
        df['is_old'] = [1 if age > 45 else 0 for age in df['Age']]
        df['is_middleaged'] = [1 if age >= 25 and age <= 45 else 0 for age in df['Age']]

        ######## 2. GENDER
        df['Gender'] = df['Gender'].replace({
            'Femal': 'Female'  # On standardise la colonne 'Gender'
        })

        ######## 3. SYMPOMS, MEDICINE, CAUSES, DISEASE
        df['Symptoms'] = df['Symptoms'].apply(lambda x: self.process_row(x))
        df['Medicine'] = df['Medicine'].apply(lambda x: self.process_row(x))
        df['Causes'] = df['Causes'].apply(lambda x: self.process_row(x))
        df['Disease'] = df['Disease'].apply(lambda x: [x])

        return df
    
    def load_options(self):
        cause_options = [
            'allergies',
            'chronic_fatigue_syndrome',
            'covid_exposure',
            'eye_strain',
            'food_poisoning',
            'infection',
            'motion_sickness',
            'other',
            'overexertion',
            'physical_exertion',
            'rheumatoid_arthritiss',
            'sciatica',
            'stress',
            'viral_infection'
        ]
        disease_options = [
            'Allergic Reaction',
            'Anxiety Disorder',
            'Arthritis',
            'COVID-19',
            'Chronic Fatigue Syndrome',
            'Gastroenteritis',
            'Herniated Disc',
            'Indigestion',
            'Motion Sickness',
            'Muscle Overuse',
            'Muscle Strain',
            'Pneumonia',
            'Tonsillitis',
            'Vision Fatigue',
            'other'
        ]
        symptom_options = [
            'abdominal_pain',
            'anxiety',
            'back_pain',
            'bloating',
            'blurred_vision',
            'chest_pain',
            'chills',
            'cough',
            'diarrhea',
            'dizziness',
            'fatigue',
            'fever',
            'headache',
            'itching',
            'joint_pain',
            'muscle_pain',
            'nausea',
            'numbness',
            'other',
            'palpitations',
            'redness',
            'shortness_of_breath',
            'sore_throat',
            'stomach_pain',
            'swelling',
            'vomiting',
            'weakness'
        ]
        return cause_options, disease_options, symptom_options