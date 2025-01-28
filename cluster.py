import schedule
import time
import requests
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# URL of the dataset endpoint
url = "https://muaadh-nazly-production.up.railway.app/get_dataset/?format=csv"

def fetch_dataset_from_url():
    """
    Fetches the dataset from the given URL and loads it into a pandas DataFrame.
    Returns:
        pd.DataFrame: The dataset as a DataFrame.
    """
    print("Starting dataset download...")
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            
            # Progress feedback
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            content = []

            # Stream and accumulate data with progress tracking
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                downloaded += len(chunk)
                content.append(chunk)
                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"Download progress: {progress:.2f}%")
            
            # Decode and load into a DataFrame
            content = b''.join(content).decode('utf-8')
            df = pd.read_csv(StringIO(content))
        
        print("Dataset successfully downloaded and loaded into a DataFrame.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the dataset: {e}")
        return None

def cluster_users(df, n_clusters=5):
    """
    Perform clustering on users based on their interaction behavior.
    Args:
        df (pd.DataFrame): The input dataset containing user interactions.
        n_clusters (int): Number of clusters for the KMeans model.
    Returns:
        pd.DataFrame: A DataFrame with user_id and their assigned cluster.
    """
    print("Starting clustering process...")
    # Aggregate data at the user level
    user_features = df.groupby('user_id').agg({
        'price': ['mean', 'sum'],  
        'product_id': 'nunique',  
        'event_type': lambda x: (x == 'purchase').sum(),  
        'brand': 'nunique',  
        'category': lambda x: x.mode()[0] 
    }).reset_index()
    print("Data aggregated by user.")
    
    # Rename columns
    user_features.columns = ['user_id', 'avg_price', 'total_spent', 'unique_products', 'num_purchases', 'unique_brands', 'favorite_category']
    print("Columns renamed for clarity.")

    # One-hot encode 'favorite_category'
    user_features = pd.get_dummies(user_features, columns=['favorite_category'], drop_first=True)
    print("One-hot encoding completed for 'favorite_category'.")

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_features.drop('user_id', axis=1))
    print("Numerical features scaled.")

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_features['cluster'] = kmeans.fit_predict(scaled_features)
    print(f"Clustering completed with {n_clusters} clusters.")

    return user_features[['user_id', 'cluster']]

def perform_clustering():
    """
    Fetches the dataset, performs clustering, and saves the results to a CSV file.
    """
    print(f"Clustering process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    df = fetch_dataset_from_url()  # Fetch the dataset
    if df is not None:  # Proceed only if the dataset was successfully fetched
        print("Performing clustering on the dataset...")
        user_clusters = cluster_users(df, n_clusters=5)  # Perform clustering
        output_file = f'user_clusters_{time.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        user_clusters.to_csv(output_file, index=False)  # Save the results
        print(f"Clustering completed and results saved to {output_file}.")
    else:
        print("Skipping clustering as the dataset could not be loaded.")

# Schedule the task to run every X minutes
X = 30  # Replace X with the desired interval in minutes
schedule.every(X).minutes.do(perform_clustering)

# Keep the script running to execute the scheduled task
print(f"Starting the scheduler to run clustering every {X} minutes...")
perform_clustering()
while True:
    schedule.run_pending()
    time.sleep(1)