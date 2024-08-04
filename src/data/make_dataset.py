import pandas as pd
data_path = "data/raw/employee_attrition.csv"
def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_csv(data_path)


    return df

if __name__ == "__main__":
    df = load_and_preprocess_data(data_path)
    print(df.head())
    
    