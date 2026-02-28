import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_condition_distributions(conditions_list, output_dir):
    """
    Analyzes the distribution of conditions extracted from the dataset
    and saves the distribution charts as images.
    
    Args:
        conditions_list (list of dict): List containing conditions for each processed track.
        output_dir (str): The directory where the plots will be saved.
    """
    if not conditions_list:
        print("No conditions data available for analysis.")
        return
        
    df = pd.DataFrame(conditions_list)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    columns_to_plot = ["COMPOSER", "VELOCITY", "DENSITY", "TEMPO", "KEY"]
    
    for col in columns_to_plot:
        if col in df.columns:
            counts = df[col].value_counts()
            
            if counts.empty:
                continue
                
            plt.figure(figsize=(12, 6))
            
            # For continuous bucket features, sort by bucket value (index)
            if col in ["VELOCITY", "DENSITY", "TEMPO"]:
                counts = counts.sort_index()
                title = f"{col} Distribution"
            # For composers, there might be too many, so we limit to top 20
            elif col == "COMPOSER" and len(counts) > 20:
                counts = counts.head(20)
                title = f"Top 20 {col} Distribution"
            else:
                title = f"{col} Distribution"
                
            counts.plot(kind="bar", color="skyblue", edgecolor="black")
            plt.title(title)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            save_path = os.path.join(plots_dir, f"{col.lower()}_distribution.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved {col} distribution plot to {save_path}")
