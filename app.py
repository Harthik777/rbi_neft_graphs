import os
from flask import Flask, render_template, request , url_for
from models import db
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import calendar
import matplotlib
from sqlalchemy import text, inspect as sql_inspect
from functools import lru_cache
import re

matplotlib.use('Agg')

# --- Database Configuration ---
user = "root"
password = "2102005"
port = 3306
host = "localhost"
new_db_name = "rbi_metric_neft"

# --- Flask App Initialization ---
app = Flask(__name__)
instance_path = os.path.join(app.instance_path)
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"mysql+pymysql://{user}:{password}@{host}/{new_db_name}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- Utility Functions ---
def get_month_name(month_num):
    try:
        num = int(month_num)
        if 1 <= num <= 12: return calendar.month_name[num]
        else: return "Invalid Month"
    except (ValueError, TypeError): return "Invalid Input"

app.jinja_env.filters['month_name'] = get_month_name

def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# --- Month name to number mapping ---
MONTH_MAP = {name.lower(): num for num, name in enumerate(calendar.month_name) if num}

# --- Define Correct Column Names (Match fileconverter.py headers) ---
# Use these constants throughout the app for consistency
COL_BANK_NAME = "`Bank Name`"
COL_SR_NO = "`Sr. No`"
COL_OUT_COUNT = "`No. Of Outward Transactions`"
COL_OUT_AMOUNT = "`Amount(Outward)`"
COL_IN_COUNT = "`No. Of Inward Transactions`"
COL_IN_AMOUNT = "`Amount(Inward)`"
# Aliases for generated columns
COL_YEAR = "`Year`"
COL_MONTH = "`Month`"

# --- Helper function to get and cache NEFT table names ---
@lru_cache(maxsize=1)
def get_neft_tables_info():
    """
    Gets the list of table names matching neft_month_year pattern
    and extracts year/month. Returns list of dicts.
    """
    neft_tables_info = []
    table_pattern = re.compile(r"neft_([a-zA-Z]+)_(\d{4})")
    try:
        # Need app context to access db.engine if called outside a request
        with app.app_context():
             inspector = sql_inspect(db.engine)
             all_tables = inspector.get_table_names()
             valid_tables = []
             for tbl in all_tables:
                 match = table_pattern.match(tbl)
                 if match:
                     month_str, year_str = match.groups()
                     month_num = MONTH_MAP.get(month_str.lower())
                     if month_num:
                          valid_tables.append({'name': tbl, 'year': int(year_str), 'month': month_num})
             # Sort chronologically primarily for predictable UNION order
             neft_tables_info = sorted(valid_tables, key=lambda x: (x['year'], x['month']))
             print(f"Found and validated NEFT tables: {[t['name'] for t in neft_tables_info]}")
    except Exception as e:
        print(f"Error inspecting database for NEFT tables: {e}")
    if not neft_tables_info:
        print("WARNING: No NEFT tables found matching pattern 'neft_month_year'.")
    return neft_tables_info

# --- Helper to build the full UNION ALL subquery string ---
def build_union_all_subquery(tables_info=None):
    """
    Builds the full UNION ALL subquery string selecting base columns
    and adding Year, Month columns derived from table names.
    """
    if tables_info is None:
        tables_info = get_neft_tables_info()
    if not tables_info:
        return None

    selects = []
    # Define ALL base columns needed by ANY query, using constants
    base_cols_select = ", ".join([
        COL_BANK_NAME,
        COL_OUT_COUNT,
        COL_OUT_AMOUNT,
        COL_IN_COUNT,
        COL_IN_AMOUNT
        # Add COL_SR_NO here if needed by any part of the app
    ])

    for table_info in tables_info:
        table_name = table_info['name']
        year_val = table_info['year']
        month_val = table_info['month']

        # Construct SELECT for this table, adding year and month as literals
        # Ensure the alias names match the constants COL_YEAR, COL_MONTH
        select_part = (
            f"SELECT {base_cols_select}, "
            f"{year_val} AS {COL_YEAR}, "
            f"{month_val} AS {COL_MONTH} "
            f"FROM `{table_name}`"
        )
        selects.append(select_part)

    # Return the combined string for use in a subquery
    return " UNION ALL ".join(selects)

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/filters')
def select_filters():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500

    # Build the common UNION ALL subquery
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build query.", 500

    # DISTINCT queries operate on the result of the subquery
    sql_distinct_banks = text(f"SELECT DISTINCT {COL_BANK_NAME} FROM ({union_subquery_sql}) AS combined_data ORDER BY {COL_BANK_NAME}")
    sql_distinct_years = text(f"SELECT DISTINCT {COL_YEAR} FROM ({union_subquery_sql}) AS combined_data ORDER BY {COL_YEAR} DESC")
    sql_distinct_months = text(f"SELECT DISTINCT {COL_MONTH} FROM ({union_subquery_sql}) AS combined_data ORDER BY {COL_MONTH} ASC")

    try:
        with db.session.begin():
            all_banks = db.session.execute(sql_distinct_banks).scalars().all()
            all_years = [int(y) for y in db.session.execute(sql_distinct_years).scalars().all() if y is not None]
            all_months_num = [int(m) for m in db.session.execute(sql_distinct_months).scalars().all() if m is not None]
    except Exception as e:
        print(f"Error fetching filter options from DB: {e}")
        # It's better to return an error template than potentially break the page
        return render_template('error.html', message=f"Database error fetching filter options: {e}"), 500
        # Or provide empty lists: all_banks, all_years, all_months_num = [], [], []

    all_months_map = {month_num: get_month_name(month_num) for month_num in all_months_num}

    selected_bank = request.args.get('bank_name', "All Banks")
    selected_year = request.args.get('year', "All Years")
    selected_month_str = request.args.get('month', "All Months")

    return render_template(
        'filters.html',
        all_banks=all_banks, all_years=all_years, all_months=all_months_map,
        selected_bank=selected_bank, selected_year=selected_year, selected_month=selected_month_str
    )

@app.route('/transactions')
def view_transactions():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500

    selected_bank = request.args.get('bank_name', "All Banks")
    selected_year = request.args.get('year', "All Years")
    selected_month_str = request.args.get('month', "All Months")

    # Build the common UNION ALL subquery
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build base query.", 500

    params = {}
    # Outer query selects needed columns from the subquery result
    sql_outer_select = f"""
        SELECT {COL_BANK_NAME}, {COL_YEAR}, {COL_MONTH},
               {COL_IN_COUNT}, {COL_OUT_COUNT},
               {COL_IN_AMOUNT}, {COL_OUT_AMOUNT}
        FROM ({union_subquery_sql}) AS combined_data
    """
    where_clauses = []
    filters_applied = False

    # Build WHERE clause using correct column constants (they refer to columns in combined_data)
    if selected_bank != "All Banks":
        where_clauses.append(f"{COL_BANK_NAME} = :bank")
        params['bank'] = selected_bank
        filters_applied = True
    if selected_year != "All Years":
        try:
            where_clauses.append(f"{COL_YEAR} = :year")
            params['year'] = int(selected_year)
            filters_applied = True
        except ValueError: pass
    if selected_month_str != "All Months":
        try:
            where_clauses.append(f"{COL_MONTH} = :month")
            params['month'] = int(selected_month_str)
            filters_applied = True
        except ValueError: pass

    if where_clauses:
        sql_query = sql_outer_select + " WHERE " + " AND ".join(where_clauses)
    else:
        sql_query = sql_outer_select

    sql_query += f" ORDER BY {COL_YEAR} DESC, {COL_MONTH} DESC, {COL_BANK_NAME} ASC"

    # Execute query for table data
    data_to_display = []
    try:
        with db.session.begin():
            results = db.session.execute(text(sql_query), params).mappings().all()
            # Process results into the format expected by the template
            for row in results:
                data_to_display.append({
                    'bank_name': row[COL_BANK_NAME.strip('`')], # Use stripped name for dict key
                    'year': row[COL_YEAR.strip('`')],
                    'month': row[COL_MONTH.strip('`')],
                    'outward_count': row[COL_OUT_COUNT.strip('`')],
                    'outward_amount': row[COL_OUT_AMOUNT.strip('`')],
                    'inward_count': row[COL_IN_COUNT.strip('`')],
                    'inward_amount': row[COL_IN_AMOUNT.strip('`')],
                })
    except Exception as e:
        print(f"Error fetching filtered transaction data: {e}")
        # Optionally return an error template
        # return render_template('error.html', message=f"Error retrieving data: {e}"), 500

    # --- Generate Conditional Filtered Graph ---
    filtered_graph_html = None
    if filters_applied and selected_bank != "All Banks" and data_to_display:
        # We already have the filtered data in 'data_to_display', no need for another SQL query.
        # Let's process this data for the graph.
        try:
            plot_data = []
            # Convert list of dicts to DataFrame for easy plotting
            plot_source_df = pd.DataFrame(data_to_display)

            # Check if necessary columns exist (they should based on the mapping above)
            if not plot_source_df.empty and all(col in plot_source_df.columns for col in ['year', 'month', 'inward_count', 'outward_count']):
                plot_source_df['total_transactions'] = plot_source_df['inward_count'] + plot_source_df['outward_count']
                # Create datetime, handle potential errors during conversion
                plot_source_df['Month_Year'] = pd.to_datetime(
                    plot_source_df['year'].astype(str) + '-' + plot_source_df['month'].astype(str) + '-01',
                    errors='coerce' # Set errors='coerce' to turn bad dates into NaT
                )
                plot_source_df.dropna(subset=['Month_Year'], inplace=True) # Remove rows with invalid dates

                # Group by month (even if only one year/month is selected, this aggregates correctly)
                plot_df_grouped = plot_source_df.groupby('Month_Year')['total_transactions'].sum().reset_index()
                plot_df_grouped = plot_df_grouped.sort_values(by='Month_Year') # Sort chronologically

                if not plot_df_grouped.empty:
                    # Create Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plot_df_grouped['Month_Year'], plot_df_grouped['total_transactions'], marker='o', linestyle='-', color='purple')
                    # ... (Set title, labels, grid etc. as before) ...
                    title = f'Monthly Transaction Volume for: {selected_bank}'
                    if selected_year != "All Years": title += f' (Year: {selected_year})'
                    if selected_month_str != "All Months": title += f' (Month: {get_month_name(selected_month_str)})'
                    ax.set_title(title)
                    ax.set_xlabel('Month-Year')
                    ax.set_ylabel('Total Transactions (Inward + Outward)')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.ticklabel_format(style='plain', axis='y')
                    plt.xticks(rotation=45)
                    fig.tight_layout()
                    filtered_graph_html = plot_to_img(fig)
        except Exception as e:
            print(f"Error generating filtered graph from processed data for {selected_bank}: {e}")

    return render_template(
        'transactions.html',
        data=data_to_display, # Use the processed data
        selected_bank=selected_bank, selected_year=selected_year, selected_month=selected_month_str,
        filtered_graph=filtered_graph_html
    )

# --- Graph Routes (Using direct SQL with UNION ALL and CORRECT columns) ---
@app.route('/graph1') # Monthly Volume
def graph1():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build query for Graph 1.", 500

    sql = text(f"""
        SELECT {COL_YEAR}, {COL_MONTH}, SUM({COL_IN_COUNT} + {COL_OUT_COUNT}) as total_transactions
        FROM ({union_subquery_sql}) AS combined_data
        GROUP BY {COL_YEAR}, {COL_MONTH}
        ORDER BY {COL_YEAR} ASC, {COL_MONTH} ASC
    """)
    try:
        with db.session.begin():
            results = db.session.execute(sql).mappings().all()
        if not results: return "No data for Graph 1.", 404
        plot_data = []
        for row in results:
           try:
               # Use stripped keys matching aliases
               dt = pd.to_datetime(f"{int(row[COL_YEAR.strip('`')])}-{int(row[COL_MONTH.strip('`')])}-01")
               plot_data.append({'Month_Year': dt, 'total_transactions': row['total_transactions']})
           except (ValueError, TypeError, KeyError): continue
        if not plot_data: return "Could not process data for Graph 1.", 500
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(plot_df['Month_Year'], plot_df['total_transactions'], marker='o', linestyle='-', color='dodgerblue')
        ax.set_title('Monthly NEFT Volume Trend (All Banks Combined)')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total Number of Transactions')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        fig.tight_layout()
        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data, graph_title="Monthly NEFT Volume")
    except Exception as e:
        print(f"Error generating graph 1: {e}")
        return "Error generating graph.", 500

@app.route('/graph2') # Monthly Value
def graph2():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build query for Graph 2.", 500

    sql = text(f"""
        SELECT {COL_YEAR}, {COL_MONTH}, SUM({COL_IN_AMOUNT} + {COL_OUT_AMOUNT}) / 10000000.0 as total_amount_cr
        FROM ({union_subquery_sql}) AS combined_data
        GROUP BY {COL_YEAR}, {COL_MONTH}
        ORDER BY {COL_YEAR} ASC, {COL_MONTH} ASC
    """)
    try:
        with db.session.begin():
            results = db.session.execute(sql).mappings().all()
        if not results: return "No data for Graph 2.", 404
        plot_data = []
        for row in results:
           try:
               dt = pd.to_datetime(f"{int(row[COL_YEAR.strip('`')])}-{int(row[COL_MONTH.strip('`')])}-01")
               amount = float(row['total_amount_cr']) if row['total_amount_cr'] is not None else 0.0
               plot_data.append({'Month_Year': dt, 'total_amount': amount})
           except (ValueError, TypeError, KeyError): continue
        if not plot_data: return "Could not process data for Graph 2.", 500
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.fill_between(plot_df['Month_Year'], plot_df['total_amount'], alpha=0.4, color='mediumseagreen')
        ax.plot(plot_df['Month_Year'], plot_df['total_amount'], marker='.', linestyle='-', color='darkgreen')
        ax.set_title('Monthly NEFT Value Trend (All Banks Combined)')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total Amount (₹ Crores)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        fig.tight_layout()
        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data, graph_title="Monthly NEFT Value")
    except Exception as e:
        print(f"Error generating graph 2: {e}")
        return "Error generating graph.", 500

@app.route('/graph3') # Top Banks by Count
def graph3():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build query for Graph 3.", 500

    sql = text(f"""
        SELECT {COL_BANK_NAME}, SUM({COL_IN_COUNT} + {COL_OUT_COUNT}) as total_transactions
        FROM ({union_subquery_sql}) AS combined_data
        GROUP BY {COL_BANK_NAME}
        ORDER BY total_transactions DESC
        LIMIT 10
    """)
    try:
        with db.session.begin():
            results = db.session.execute(sql).mappings().all()
        if not results: return "No data for Graph 3.", 404
        # Use stripped key 'Bank Name'
        banks = [row[COL_BANK_NAME.strip('`')] for row in results]
        transactions = [row['total_transactions'] for row in results]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(banks[::-1], transactions[::-1], color='skyblue')
        ax.set_title('Top 10 Banks by Total Transaction Count')
        ax.set_xlabel('Total Number of Transactions')
        ax.ticklabel_format(style='plain', axis='x')
        fig.tight_layout()
        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data, graph_title="Top Banks by Transaction Count")
    except Exception as e:
        print(f"Error generating graph 3: {e}")
        return "Error generating graph.", 500

@app.route('/graph4') # Top Banks by Amount
def graph4():
    tables_info = get_neft_tables_info()
    if not tables_info: return "Error: No NEFT data tables found.", 500
    union_subquery_sql = build_union_all_subquery(tables_info=tables_info)
    if not union_subquery_sql: return "Error: Could not build query for Graph 4.", 500

    sql = text(f"""
        SELECT {COL_BANK_NAME}, SUM({COL_IN_AMOUNT} + {COL_OUT_AMOUNT}) / 10000000.0 as total_amount_cr
        FROM ({union_subquery_sql}) AS combined_data
        GROUP BY {COL_BANK_NAME}
        ORDER BY total_amount_cr DESC
        LIMIT 10
    """)
    try:
        with db.session.begin():
            results = db.session.execute(sql).mappings().all()
        if not results: return "No data for Graph 4.", 404
        banks = [row[COL_BANK_NAME.strip('`')] for row in results]
        amounts = [float(row['total_amount_cr']) if row['total_amount_cr'] is not None else 0.0 for row in results]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(banks, amounts, color='lightcoral')
        ax.set_title('Top 10 Banks by Total NEFT Amount')
        ax.set_ylabel('Total Amount (₹ Crores)')
        ax.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=60, ha='right')
        fig.tight_layout()
        img_data = plot_to_img(fig)
        return render_template('graph.html', img_data=img_data, graph_title="Top Banks by Transaction Value")
    except Exception as e:
        print(f"Error generating graph 4: {e}")
        return "Error generating graph.", 500


# --- Main execution block ---
if __name__ == '__main__':
    try:
        # Call get_neft_tables_info() once on startup to cache the list
        # and perform an initial check. Requires app context.
        with app.app_context():
            get_neft_tables_info()
    except Exception as e:
        print(f"CRITICAL: Failed initial database connection/inspection: {e}")
    print("Starting Flask application...")
    app.run(debug=True)