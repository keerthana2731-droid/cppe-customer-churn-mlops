import pandas as pd
import numpy as np
import os

def generate_churn_dataset(n_samples=1000, output_path="data/churn_data.csv"):
    np.random.seed(42)
    
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 110.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 15, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    contract_types = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]) # 0: Month-to-month, 1: One year, 2: Two year
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35]) # 0: No, 1: Yes
    online_security = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    
    # Calculate churn probability based on features
    churn_prob = (
        0.35 
        - 0.005 * tenure 
        + 0.004 * (monthly_charges - 50) 
        - 0.20 * contract_types 
        - 0.15 * tech_support
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.90)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "tech_support": tech_support,
        "online_security": online_security,
        "churn": churn
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} customer records in {output_path}")
    print(f"Churn rate: {df['churn'].mean()*100:.1f}%")

if __name__ == "__main__":
    generate_churn_dataset()
