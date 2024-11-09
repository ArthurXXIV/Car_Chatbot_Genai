import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data\output.csv')

# Step 1: Remove hyphens from 'Brand'
df['Brand'] = df['Brand'].str.replace('-cars', '', regex=False)
df['Brand'] = df['Brand'].str.replace('-', ' ', regex=False)

# Step 2: Remove the brand name (either one or both words) from 'Car'
def clean_car(row):
    brand_words = row['Brand'].split()  # Split brand into words
    car_name = row['Car']
    
    for word in brand_words:
        if word.lower() in car_name.lower():  # Case-insensitive comparison
            car_name = car_name.lower().replace(word.lower(), '').strip()
    
    return car_name.title()  # Return the cleaned car name with proper capitalization

df['Car'] = df.apply(clean_car, axis=1)

# Step 3: Lowercase both 'Car' and 'Variant', and remove the car name from 'Variant'
def clean_variant(row):
    car_name = row['Car'].lower()
    variant_name = row['Variant'].lower() if pd.notna(row['Variant']) else ''
    
    if car_name in variant_name:
        variant_name = variant_name.replace(car_name, '').strip()
    
    return variant_name.title()  # Return the cleaned variant with proper capitalization

df['Variant'] = df.apply(clean_variant, axis=1)


# List of mandatory and optional columns
mandatory_columns = ['Brand', 'Car', 'Variant', 'Description', 'Price', 'Fuel Type', 'Mileage', 'Transmission', 'Engine', 'Display']
optional_columns = [
    'Sunroof / Moonroof', 'Reverse Camera', 'Touchscreen Display', 'Alloy Wheels', 'Music System', 'Dashcam', 
    'Rear AC', 'Central Locking', 'Cruise Control', 'Hill Hold Control', 'Ventilated Seats', 'Wireless Charger',
    'Instrument Cluster', 'Adjustable ORVMs', 'Integrated (in-dash) Music System', 'Speakers'
]

# Convert '-' to 'no' for all columns
df.replace('-', 'no', inplace=True)

# Drop rows with NaN values in 'Variant' and 'Price'
df = df.dropna(subset=['Variant', 'Price'])

# Replace NaN values in other mandatory columns with None
df[mandatory_columns] = df[mandatory_columns].apply(lambda x: x.where(pd.notna(x), None))

# Helper function to convert price strings with 'Lakh' and 'Crore' to numeric
def convert_to_numeric(price_str):
    if 'Lakh' in price_str:
        return round(float(price_str.replace('Lakh', '').strip()) * 1e5)
    else:
        return round(float(price_str.replace('Crore', '').strip()) * 1e7)

# Clean 'Price' column manually and handle ranges
def clean_price(price_str):
    """
    Cleans and converts price strings to float. Handles 'Rs.', 'Lakh', 'Crore', ranges, and other potential issues.
    """
    try:
        # Remove unnecessary text like 'Estimated Price'
        price_str = price_str.replace('Estimated Price', '').strip()
        
        # Remove 'Rs.' and commas
        price_str = price_str.replace('Rs.', '').replace(',', '')
        
        # Handle ranges (e.g., Rs. 12.00 - 18.00 Lakh)
        if '-' in price_str:
            price_range = price_str.split('-')
            low = price_range[0].strip()
            high = price_range[1].strip()
            
            # Convert low and high to numeric (consider Lakh/Crore)
            low_val = convert_to_numeric(low)
            high_val = convert_to_numeric(high)
            
            # Return the median of the range
            return round((low_val + high_val) / 2)
        
        # If no range, convert the single price
        else:
            return convert_to_numeric(price_str)
    
    except ValueError:
        return np.nan  # Return NaN for any conversion errors

# Apply the cleaning function to the 'Price' column
df['Price'] = df['Price'].apply(clean_price)

# Drop rows where 'Price' could not be converted to a valid float (i.e., NaN values)
df = df.dropna(subset=['Price'])

# Check which optional columns are present in the dataset
present_optional_columns = [col for col in optional_columns if col in df.columns]

# Replace NaN values in present optional columns with 'no'
df[present_optional_columns] = df[present_optional_columns].apply(lambda x: x.where(pd.notna(x), 'no'))

# Function to create a combined description, ignoring 'None' values
def combine_description(row):
    # Combine mandatory columns (but show 'Description' without column name)
    combined_info = ' | '.join([f"{col}: {row[col]}" if col != 'Description' and row[col] is not None else str(row[col]) for col in mandatory_columns if row[col] is not None])

    # Add optional columns if they are present and not NaN, with format 'column: info'
    optional_info = ' | '.join([f"{col}: {row[col]}" for col in present_optional_columns if row[col] is not None])
    
    # Combine both parts
    if optional_info:
        combined_info += ' | ' + optional_info
    return combined_info

# Create the 'Combined Description' column
df['Combined Description'] = df.apply(combine_description, axis=1)

# Keep only the necessary columns (mandatory, optional, and 'Combined Description')
columns_to_keep = mandatory_columns + [col for col in optional_columns if col in df.columns] + ['Combined Description']
new_df = df[columns_to_keep]

# Display the resulting dataframe
new_df.to_csv('data/Cleaned_data.csv')
