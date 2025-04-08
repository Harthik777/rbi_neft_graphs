import os # Import os
from flask import Flask, render_template, request # Added request
from models import db
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import calendar # Import calendar for month names
import matplotlib
from decimal import Decimal # Import Decimal

# Use Agg backend for matplotlib
matplotlib.use('Agg')

user = "root"
password = "2102005"
port = 3306
host = "localhost"
new_db_name = "rbi_metric_neft"

app = Flask(__name__)
# Ensure the instance folder exists if using SQLite default fallback
instance_path = os.path.join(app.instance_path)
os.makedirs(instance_path, exist_ok=True)

# Use environment variable or default to your MySQL connection
# IMPORTANT: For security, avoid hardcoding credentials. Use environment variables in production.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"mysql+pymysql://{user}:{password}@{host}/{new_db_name}"
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- Utility to get month name ---
def get_month_name(month_num):
    try:
        # Ensure month_num is treated as an integer
        num = int(month_num)
        if 1 <= num <= 12:
            return calendar.month_name[num]
        else:
            return "Invalid Month"
    except (ValueError, TypeError):
        return "Invalid Input"

# Register the custom filter with Jinja environment
app.jinja_env.filters['month_name'] = get_month_name

# --- Preload Data (Improved Robustness) ---
with app.app_context():
    def load_all_neft_data():
        all_data = []
        try: # Wrap the whole loading process in a try block
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
        except Exception as e:
            print(f"Error connecting to DB or inspecting tables: {e}")
            return pd.DataFrame() # Return empty DataFrame on DB connection error

        neft_tables = [t for t in tables if t.startswith('neft_')]
        print(f"Found NEFT tables: {neft_tables}") # Debug print

        month_lookup = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        for table_name in neft_tables:
            try:
                parts = table_name.split('_')
                if len(parts) < 3:
                    print(f"Skipping table with unexpected name format: {table_name}")
                    continue
                month_str = parts[1].lower()
                # Ensure the year part is digits before converting
                if not parts[2].isdigit():
                     print(f"Skipping table with non-integer year part: {table_name}")
                     continue
                year = int(parts[2])

                if month_str not in month_lookup:
                    print(f"Skipping table with invalid month: {table_name}")
                    continue # Skip instead of crashing
                month_num = month_lookup[month_str]

                table = db.Table(table_name, db.metadata, autoload_with=db.engine)
                query = db.session.query(table)
                # Fetch rows safely
                rows = [dict(row._mapping) for row in query.all()]

                for row in rows:
                    row['month'] = month_num
                    row['year'] = year
                all_data.extend(rows)
            except Exception as e:
                print(f"Error processing table {table_name}: {e}")
                # Decide whether to continue or stop based on the error
                continue # Continue processing other tables

        if not all_data:
             print("Warning: No data loaded from NEFT tables.")
             return pd.DataFrame() # Return empty DataFrame if no data

        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} rows into initial DataFrame.") # Debug print

        # Define expected original column names
        rename_map = {
            'Bank Name': 'bank_name',
            'No. Of Outward Transactions': 'outward_count',
            'Amount(Outward)': 'outward_amount',
            'No. Of Inward Transactions': 'inward_count',
            'Amount(Inward)': 'inward_amount'
        }
        # Create rename map only with columns that actually exist in the DataFrame
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename_map, inplace=True)
        print(f"Columns after rename: {df.columns.tolist()}") # Debug print

        # --- Data Type Conversion and Cleaning ---
        # Columns expected to be numeric
        numeric_cols = ['outward_count', 'outward_amount', 'inward_count', 'inward_amount']
        # Ensure base columns exist before processing
        ensure_cols_exist = ['year', 'month', 'bank_name'] + numeric_cols

        for col in ensure_cols_exist:
             if col not in df.columns:
                  print(f"Warning: Column '{col}' not found in loaded data. Creating it.")
                  if col in numeric_cols:
                      df[col] = 0 # Default numeric to 0
                  elif col == 'bank_name':
                      df[col] = 'Unknown' # Default bank name
                  else:
                      df[col] = None # Default others to None or appropriate default


        # Convert amount columns (handle Decimal, strings with commas, etc.)
        amount_cols = ['outward_amount', 'inward_amount']
        for col in amount_cols:
            if col in df.columns:
                # Convert Decimal objects to float first if they exist
                if any(isinstance(x, Decimal) for x in df[col]):
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                # Convert strings (potentially with commas) or other types to numeric, coercing errors
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                df[col].fillna(0.0, inplace=True) # Fill any conversion errors or NaNs with 0.0

        # Convert count columns (handle potential non-numeric safely)
        count_cols = ['outward_count', 'inward_count']
        for col in count_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(0, inplace=True)
                df[col] = df[col].astype(int) # Convert counts to integer

        # Ensure 'year' and 'month' are integers
        if 'year' in df.columns: df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        if 'month' in df.columns: df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        # Ensure bank name is string
        if 'bank_name' in df.columns: df['bank_name'] = df['bank_name'].astype(str).fillna('Unknown')

        print(f"DataFrame dtypes after cleaning:\n{df.dtypes}") # Debug print
        print(f"DataFrame head after cleaning:\n{df.head()}") # Debug print
        return df

    global_df = load_all_neft_data()
    if global_df.empty:
        print("CRITICAL WARNING: global_df is empty after loading. Check DB connection and table data.")


# --- Modified Home Route ---
@app.route('/')
def home():
    # Ensure global_df exists and is not empty before proceeding
    if 'global_df' not in globals() or global_df.empty:
         # Render a specific error page or a simple message
         return "Error: NEFT data could not be loaded. Please check the application logs.", 500

    # Make a copy to avoid modifying the global DataFrame
    df_filtered = global_df.copy()

    # Get filter values from request arguments (URL parameters)
    selected_bank = request.args.get('bank_name', "All Banks")
    selected_year = request.args.get('year', "All Years")
    # Month number comes as string from request.args
    selected_month_str = request.args.get('month', "All Months")

    # --- Apply filters using Pandas boolean indexing ---
    conditions = pd.Series([True] * len(df_filtered)) # Start with all true

    if selected_bank != "All Banks" and 'bank_name' in df_filtered.columns:
        conditions &= (df_filtered['bank_name'] == selected_bank)

    if selected_year != "All Years" and 'year' in df_filtered.columns:
        try:
            conditions &= (df_filtered['year'] == int(selected_year))
        except ValueError:
            pass # Ignore non-integer year filter

    if selected_month_str != "All Months" and 'month' in df_filtered.columns:
        try:
            conditions &= (df_filtered['month'] == int(selected_month_str))
        except ValueError:
            pass # Ignore non-integer month filter

    # Apply the combined conditions
    df_filtered = df_filtered[conditions]

    # Prepare data for the template: Convert filtered DataFrame to list of dictionaries
    data_to_display = df_filtered.to_dict('records')

    # Get unique values for filter dropdowns from the original global_df
    # Add checks to ensure columns exist before calling unique()
    all_banks = sorted(global_df['bank_name'].unique().tolist()) if 'bank_name' in global_df else []
    all_years = sorted(global_df['year'].unique().tolist(), reverse=True) if 'year' in global_df else []
    # Get unique month numbers and map to names for display
    all_months_num = sorted(global_df['month'].unique().tolist()) if 'month' in global_df else []
    # Create dict {month_num: month_name}
    all_months_map = {month_num: get_month_name(month_num) for month_num in all_months_num}

    return render_template(
        'home.html',
        data=data_to_display,
        all_banks=all_banks,
        all_years=all_years,
        all_months=all_months_map, # Pass the {num: name} dict
        # Pass selected values back to keep filters selected
        selected_bank=selected_bank,
        selected_year=selected_year,
        selected_month=selected_month_str # Pass back the string version
    )


# --- Graph routes (Add checks for global_df and columns) ---
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

@app.route('/graph1')
def graph1():
    required_cols = ['year', 'month', 'inward_count', 'outward_count']
    if 'global_df' not in globals() or global_df.empty or not all(col in global_df.columns for col in required_cols):
        return "Error: Data for Graph 1 is unavailable or incomplete.", 500
    try:
        df = global_df.copy()
        df['Month_Year'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df_grouped = df.groupby('Month_Year').agg({'inward_count': 'sum', 'outward_count': 'sum'}).reset_index()
        df_grouped['total_transactions'] = df_grouped['inward_count'] + df_grouped['outward_count']

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(df_grouped['Month_Year'], df_grouped['total_transactions'], marker='o')
        ax.set_title('Monthly NEFT Volume Trend (All Banks Combined)')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total # of Transactions')
        ax.grid(True)
        ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation

        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data)
    except Exception as e:
        print(f"Error generating graph 1: {e}")
        return "Error generating graph.", 500


@app.route('/graph2')
def graph2():
    required_cols = ['year', 'month', 'inward_amount', 'outward_amount']
    if 'global_df' not in globals() or global_df.empty or not all(col in global_df.columns for col in required_cols):
         return "Error: Data for Graph 2 is unavailable or incomplete.", 500
    try:
        df = global_df.copy()
        df['Month_Year'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df_grouped = df.groupby('Month_Year').agg({'inward_amount': 'sum', 'outward_amount': 'sum'}).reset_index()
        df_grouped['total_amount'] = df_grouped['inward_amount'] + df_grouped['outward_amount']

        fig, ax = plt.subplots(figsize=(10,6))
        ax.fill_between(df_grouped['Month_Year'], df_grouped['total_amount'], alpha=0.5)
        ax.plot(df_grouped['Month_Year'], df_grouped['total_amount'], marker='o')
        ax.set_title('Monthly NEFT Value Trend (All Banks Combined)')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total Amount')
        ax.grid(True)
        ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation

        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data)
    except Exception as e:
        print(f"Error generating graph 2: {e}")
        return "Error generating graph.", 500


@app.route('/graph3')
def graph3():
    required_cols = ['bank_name', 'inward_count', 'outward_count']
    if 'global_df' not in globals() or global_df.empty or not all(col in global_df.columns for col in required_cols):
         return "Error: Data for Graph 3 is unavailable or incomplete.", 500
    try:
        df = global_df.copy()
        # Ensure counts are numeric before summing
        df['inward_count'] = pd.to_numeric(df['inward_count'], errors='coerce').fillna(0)
        df['outward_count'] = pd.to_numeric(df['outward_count'], errors='coerce').fillna(0)
        df['total_transactions'] = df['inward_count'] + df['outward_count']

        df_grouped = df.groupby('bank_name').agg({'total_transactions': 'sum'}).reset_index()
        df_top10 = df_grouped.sort_values(by='total_transactions', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(df_top10['bank_name'], df_top10['total_transactions'], color='skyblue')
        ax.set_title('Top 10 Banks by Total Transactions')
        ax.set_xlabel('Total Transactions')
        ax.invert_yaxis()
        ax.ticklabel_format(style='plain', axis='x') # Prevent scientific notation

        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data)
    except Exception as e:
        print(f"Error generating graph 3: {e}")
        return "Error generating graph.", 500

@app.route('/graph4')
def graph4():
    required_cols = ['bank_name', 'inward_amount', 'outward_amount']
    if 'global_df' not in globals() or global_df.empty or not all(col in global_df.columns for col in required_cols):
         return "Error: Data for Graph 4 is unavailable or incomplete.", 500
    try:
        df = global_df.copy()
        # Ensure amounts are numeric before summing
        df['inward_amount'] = pd.to_numeric(df['inward_amount'], errors='coerce').fillna(0.0)
        df['outward_amount'] = pd.to_numeric(df['outward_amount'], errors='coerce').fillna(0.0)
        df['total_amount'] = df['inward_amount'] + df['outward_amount']

        df_grouped = df.groupby('bank_name').agg({'total_amount': 'sum'}).reset_index()
        df_top10 = df_grouped.sort_values(by='total_amount', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(df_top10['bank_name'], df_top10['total_amount'], color='orange')
        ax.set_title('Top 10 Banks by Total NEFT Amount')
        ax.set_ylabel('Total Amount')
        ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
        plt.xticks(rotation=45, ha='right') # Correct way to rotate xticks with matplotlib Axes object
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data)
    except Exception as e:
        print(f"Error generating graph 4: {e}")
        return "Error generating graph.", 500


if __name__ == '__main__':
    # Add check after loading data
    if 'global_df' not in globals() or global_df.empty:
       print("*"*20)
       print("WARNING: No NEFT data was loaded. The application might not function correctly.")
       print("*"*20)
    # Consider adding db.create_all() if models were defined and needed tables
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)