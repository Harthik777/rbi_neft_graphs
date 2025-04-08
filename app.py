from flask import Flask, render_template
from models import db
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import calendar
import matplotlib

# Use Agg backend for matplotlib
matplotlib.use('Agg')

user = "root"
password = "2102005"
port = 3306
host = "localhost"
new_db_name = "rbi_metric_neft"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{user}:{password}@{host}/{new_db_name}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Preload Data
with app.app_context():
    def load_all_neft_data():
        all_data = []
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        neft_tables = [t for t in tables if t.startswith('neft_')]

        month_lookup = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        for table_name in neft_tables:
            parts = table_name.split('_')
            month_str = parts[1].lower()
            year = int(parts[2])

            if month_str not in month_lookup:
                raise Exception(f"Invalid month abbreviation '{month_str}' found in table name: {table_name}")
            month_num = month_lookup[month_str]

            table = db.Table(table_name, db.metadata, autoload_with=db.engine)
            query = db.session.query(table)
            rows = [dict(row._mapping) for row in query.all()]

            for row in rows:
                row['month'] = month_num
                row['year'] = year
            all_data.extend(rows)

        df = pd.DataFrame(all_data)

        # Rename columns to simpler names
        df.rename(columns={
            'Bank Name': 'bank_name',
            'No. Of Outward Transactions': 'outward_count',
            'Amount(Outward)': 'outward_amount',
            'No. Of Inward Transactions': 'inward_count',
            'Amount(Inward)': 'inward_amount'
        }, inplace=True)

        # 🔥 Convert Decimals to floats
        float_cols = ['outward_amount', 'inward_amount']
        for col in float_cols:
            df[col] = df[col].astype(float)

        return df

    global_df = load_all_neft_data()

def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/graph1')
def graph1():
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

    img_data = plot_to_img(fig)
    return render_template('graph.html', img_data=img_data)

@app.route('/graph2')
def graph2():
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

    img_data = plot_to_img(fig)
    return render_template('graph.html', img_data=img_data)

@app.route('/graph3')
def graph3():
    df = global_df.copy()
    df['total_transactions'] = df['inward_count'] + df['outward_count']
    df_grouped = df.groupby('bank_name').agg({'total_transactions': 'sum'}).reset_index()
    df_top10 = df_grouped.sort_values(by='total_transactions', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(df_top10['bank_name'], df_top10['total_transactions'], color='skyblue')
    ax.set_title('Top 10 Banks by Total Transactions')
    ax.set_xlabel('Total Transactions')
    ax.invert_yaxis()

    img_data = plot_to_img(fig)
    return render_template('graph.html', img_data=img_data)

@app.route('/graph4')
def graph4():
    df = global_df.copy()
    df['total_amount'] = df['inward_amount'] + df['outward_amount']
    df_grouped = df.groupby('bank_name').agg({'total_amount': 'sum'}).reset_index()
    df_top10 = df_grouped.sort_values(by='total_amount', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(df_top10['bank_name'], df_top10['total_amount'], color='orange')
    ax.set_title('Top 10 Banks by Total NEFT Amount')
    ax.set_ylabel('Total Amount')
    ax.set_xticklabels(df_top10['bank_name'], rotation=45, ha='right')

    img_data = plot_to_img(fig)
    return render_template('graph.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
