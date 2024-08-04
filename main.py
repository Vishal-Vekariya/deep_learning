from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import deep_learning, best_rate, activation_function
from src.models.predict_model import evaluate_model
from src.visualization.visualize import semi_log,loss_cruve
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
  
    
    data_path = "data/raw/employee_attrition.csv"
    df = load_and_preprocess_data(data_path)
    
    Y,X,X_scaled = create_dummy_vars(df)
    
    comp,eva = deep_learning( Y,X,X_scaled)
    comp
    print(eva)
    
    history = best_rate( Y,X,X_scaled)
   
    semi_log(history)
    
    
    y_test, pred, history1 = activation_function( Y,X,X_scaled)
    acc = evaluate_model (y_test, pred)
    
    print(acc)
    loss_cruve(history1)
    
    