import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import altair as alt
from datetime import datetime, timezone, timedelta
import ftplib
import io

# =================================================================================
# --- PAGE CONFIGURATION ---
# =================================================================================
st.set_page_config(
    page_title="Supply Chain & Finance Dashboard",
    page_icon="📦",
    layout="wide"
)


# =================================================================================
# --- HELPER FUNCTIONS ---
# =================================================================================

def format_indian_currency(num):
    """Correctly formats a number into the Indian currency system for tables."""
    if pd.isna(num):
        return "₹ 0.00"
    s = f"{float(num):,.2f}"
    if '.' in s:
        integer_part, decimal_part = s.split('.')
    else:
        integer_part, decimal_part = s, "00"

    integer_part = integer_part.replace(',', '')
    last_three = integer_part[-3:]
    other_digits = integer_part[:-3]

    if other_digits:
        formatted_other_digits = ','.join(
            [other_digits[max(0, i - 2):i] for i in range(len(other_digits), 0, -2)][::-1]
        )
        return f"₹ {formatted_other_digits},{last_three}.{decimal_part}"
    else:
        return f"₹ {last_three}.{decimal_part}"

def format_indian_currency_kpi(num):
    """Formats a number into a compact Indian currency string for KPIs (Cr, L, K)."""
    if pd.isna(num):
        return "₹ 0"
    num = float(num)
    if num >= 1_00_00_000:
        return f"₹ {num / 1_00_00_000:.2f} Cr"
    elif num >= 1_00_000:
        return f"₹ {num / 1_00_000:.2f} L"
    elif num >= 1_000:
        return f"₹ {num / 1_000:.2f} K"
    else:
        return f"₹ {num:,.2f}"

def format_indian_number(num):
    """Formats a number into the Indian numbering system for labels."""
    if pd.isna(num):
        return "0"
    s = str(int(num))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    other_digits = s[:-3]
    formatted_other_digits = ','.join([other_digits[max(0, i - 2):i] for i in range(len(other_digits), 0, -2)][::-1])
    return f"{formatted_other_digits},{last_three}"


@st.cache_data(ttl=300)
def load_data_from_ftp(_ftp_paths):
    """
    Connects to an FTP server, downloads files, gets their last modified times,
    and loads them into pandas DataFrames.
    Returns a tuple of (data_frames, file_timestamps).
    """
    data_frames = {}
    file_timestamps = {}
    ftp_host = st.secrets["ftp"]["host"]
    ftp_user = st.secrets["ftp"]["user"]
    ftp_pass = st.secrets["ftp"]["password"]

    try:
        with ftplib.FTP(ftp_host, ftp_user, ftp_pass) as ftp:
            st.info("🔌 Successfully connected to FTP server...")

            for name, path_on_ftp in _ftp_paths.items():
                try:
                    mdtm_response = ftp.sendcmd(f'MDTM {path_on_ftp}')
                    timestamp_str = mdtm_response.split(' ')[1].strip()
                    naive_dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    utc_dt = naive_dt.replace(tzinfo=timezone.utc)
                    ist_tz = timezone(timedelta(hours=5, minutes=30))
                    mod_time_ist = utc_dt.astimezone(ist_tz)
                    file_timestamps[name] = mod_time_ist

                except Exception as e:
                    file_timestamps[name] = None
                    st.warning(f"Could not retrieve last modified time for '{name}': {e}")

                in_memory_file = io.BytesIO()
                ftp.retrbinary(f'RETR {path_on_ftp}', in_memory_file.write)
                in_memory_file.seek(0)

                df = pd.read_parquet(in_memory_file)
                df.columns = df.columns.str.strip()
                data_frames[name] = df

        st.success("✅ All data files loaded successfully!")
        return data_frames, file_timestamps

    except ftplib.all_errors as e:
        st.error(f"FTP Error: {e}. Please check your credentials and file paths in secrets.toml.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return None, None


def display_last_refreshed(file_keys, timestamps_dict):
    """
    Finds the latest timestamp from a list of files and displays it in the sidebar.
    """
    relevant_timestamps = [timestamps_dict.get(key) for key in file_keys if timestamps_dict.get(key)]
    if not relevant_timestamps:
        st.sidebar.markdown("Last refreshed: *Unknown*")
        return

    latest_timestamp = max(relevant_timestamps)
    formatted_timestamp = latest_timestamp.strftime('%d-%b-%Y, %I:%M %p')
    st.sidebar.markdown(f"Last refreshed: **{formatted_timestamp}**")


def create_grouped_bar_chart(df, title):
    """Creates a grouped (clustered) Altair bar chart for dispatch and delivery timelines WITH data labels."""
    dispatch_data = df['Dispatch Category'].value_counts().reset_index()
    dispatch_data.columns = ['Category', 'count']
    dispatch_data['Timeline'] = 'Dispatch'

    delivery_data = df['Delivery Category'].value_counts().reset_index()
    delivery_data.columns = ['Category', 'count']
    delivery_data['Timeline'] = 'Delivery'

    combined_data = pd.concat([dispatch_data, delivery_data])

    category_order = ['0-1 Day', '2 Days', '3 Days', '4-5 Days', 'More than 5 Days', 'Invalid Data']

    bars = alt.Chart(combined_data).mark_bar().encode(
        x=alt.X('Category:N', axis=alt.Axis(title=None, labelAngle=0), sort=category_order),
        y=alt.Y('count:Q', axis=alt.Axis(title='Number of Orders')),
        color=alt.Color('Timeline:N',
                        scale=alt.Scale(domain=['Dispatch', 'Delivery'],
                                        range=['#1f77b4', '#ff7f0e']),
                        legend=alt.Legend(title="Timeline")),
        xOffset='Timeline:N',
        tooltip=[
            alt.Tooltip('Category', title='Category'),
            alt.Tooltip('Timeline', title='Timeline'),
            alt.Tooltip('count', title='Orders')
        ]
    )

    text = bars.mark_text(
        align='center',
        baseline='middle',
        dy=-10
    ).encode(
        text='count:Q'
    )

    chart = (bars + text).properties(
        title=title
    ).configure_view(
        stroke='transparent'
    ).configure_axis(
        grid=False
    ).configure_legend(
        orient='top'
    )
    return chart

def create_combo_chart(df, x_col, bar_col, line_col, bar_text, line_text, title):
    """Creates a Plotly combination chart with bars and a line on a secondary y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[bar_col] / 1_00_000,
            name='Stock Value',
            marker_color='mediumpurple',
            text=bar_text,
            textposition='inside',
            insidetextanchor='middle',
            textfont_color='white',
            textangle=0,
            hovertemplate=f'<b>%{{x}}</b><br>Stock Value: ₹ %{{y:.2f}} L<extra></extra>'
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[line_col],
            name='Qty in Cases',
            mode='lines+markers+text',
            text=line_text,
            textposition='top center',
            marker_color='darkslateblue',
            hovertemplate=f'<b>%{{x}}</b><br>Qty in Cases: %{{y:,.0f}}<extra></extra>'
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text="<b>Stock Value (in ₹ Lakhs)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Qty in Cases</b>", tickformat=",")

    return fig

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')


# =================================================================================
# --- MAIN APPLICATION ---
# =================================================================================

st.title("Supply Chain & Finance Analytics Dashboard")

ftp_file_paths = st.secrets["ftp"]["paths"]
data, file_timestamps = load_data_from_ftp(ftp_file_paths)


if data and file_timestamps:
    # =================================================================================
    # --- DATA PREPARATION (Done once upfront for all pages) ---
    # =================================================================================

    # --- Stock Analysis Data ---
    columns_to_merge = ['Item No.', 'Classification', 'Remark', 'Vendor']
    if 'Item Description' in data['skulist'].columns:
        columns_to_merge.append('Item Description')
    else:
        st.warning("Warning: 'Item Description' column not found in skulist data. Using 'Item No.' as fallback for SKU-level views.")

    stock_enriched = pd.merge(
        data['stockav'],
        data['skulist'][columns_to_merge],
        left_on='ItemCode',
        right_on='Item No.',
        how='left'
    )
    stock_enriched = pd.merge(stock_enriched, data['dsmmaster'], left_on='WhsCode', right_on='WH Code', how='left')

    tn_depots = ["CHN_N", "ERD", "TRI", "TUTICOR"]
    Damage_goods =["DBAN","DBHIWA","DCHN_N","DCUT","DEcom","DERD","DHYD_New","DIND","DKOL","DNAG","DTRI","DTUTICOR"]
    transit=['TCHN','TCHN_N','TEcom','THYD_New','TIND','TTRI','TTUTICOR','TTUTMFD']

    conditions = [
        stock_enriched['WhsCode'].isin(tn_depots),
        stock_enriched['WhsCode'].isin(Damage_goods),
        stock_enriched['WhsCode'].isin(transit)
    ]
    choices = ['TN', 'Damage_goods', 'transit']
    stock_enriched['State_Group'] = np.select(conditions, choices, default='OTN')

    numeric_cols_stock = ['Stock Value', 'Qty in Cases']
    for col in numeric_cols_stock:
        if col in stock_enriched.columns:
            stock_enriched[col] = pd.to_numeric(
                stock_enriched[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            ).fillna(0)

    categorical_cols_stock = ['RSM/ DSM', 'Classification', 'Prod Cat', 'Remark', 'WhsCode', 'Vendor']
    if 'ItemName' in stock_enriched.columns:
        categorical_cols_stock.append('ItemName')

    for col in categorical_cols_stock:
        if col in stock_enriched.columns:
            stock_enriched[col].fillna("Unknown", inplace=True)

    # --- Open PO Data ---
    open_po_data = data['openpo'].copy()
    open_po_data['Qty in Cases'] = pd.to_numeric(
        open_po_data['Qty in Cases'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    ).fillna(0)
    open_po_data['Net Rate'] = pd.to_numeric(
        open_po_data['Net Rate'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    ).fillna(0)
    open_po_data['CardName'].fillna("Unknown Vendor", inplace=True)
    open_po_data['Open Order Value'] = open_po_data['Net Rate']
    if 'DocDate' in open_po_data.columns:
        open_po_data['DocDate'] = pd.to_datetime(open_po_data['DocDate'], errors='coerce', dayfirst=True)
        current_date = pd.Timestamp.now().normalize()
        open_po_data['Open PO Aging (Days)'] = (current_date - open_po_data['DocDate']).dt.days
        open_po_data['Open PO Aging (Days)'].fillna(0, inplace=True)
        open_po_data['Open PO Aging (Days)'] = open_po_data['Open PO Aging (Days)'].astype(int)

    # --- Stock Aging Data ---
    stock_aging_data = data['stockaging'].copy()
    numeric_cols_aging = [
        'MRP', 'NumInsale', 'OnHand', 'StockValue',
        'TotalQty', 'TotalValue', '0-15Qty', '0-15Value', '16-30Qty', '16-30Value',
        '31-60Qty', '31-60Value', '61-90Qty', '61-90Value', '91-180Qty', '91-180Value',
        '181-360Qty', '181-360Value', '361-720Qty', '361-720Value', '721+Qty', '721+DaysValue'
    ]
    for col in numeric_cols_aging:
        if col in stock_aging_data.columns:
            stock_aging_data[col] = pd.to_numeric(
                stock_aging_data[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
    stock_aging_data[numeric_cols_aging] = stock_aging_data[numeric_cols_aging].fillna(0)

    for col in ['Brand', 'State', 'WhsCode', 'ProductGroup']:
        if col in stock_aging_data.columns:
            stock_aging_data[col].fillna("Unknown", inplace=True)

    # --- Dispatch & Delivery Data ---
    db_master_df = data['dbmaster'].copy()
    dd_df = data['dd'].copy()
    depot_mapping_df = data['depot_mapping'].copy()

    if 'CardCode' in db_master_df.columns:
        db_master_df['CardCode'] = db_master_df['CardCode'].str.strip()
    if 'CardCode' in dd_df.columns:
        dd_df['CardCode'] = dd_df['CardCode'].str.strip()

    if 'Qty in Cases' in dd_df.columns:
        dd_df['Qty in Cases'] = pd.to_numeric(
            dd_df['Qty in Cases'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        dd_df['Qty in Cases'] = 0
        st.warning("Warning: 'Qty in Cases' column not found in Dispatch & Delivery data. Case-based metrics will be zero.")

    db_master_df = db_master_df.loc[:, ~db_master_df.columns.duplicated()]
    db_master_df = pd.merge(db_master_df, depot_mapping_df, left_on='U_Depot', right_on='Depot to Bill', how='left')
    db_master_df['Final Depot to Bill'] = db_master_df['Updated_depot_to_bill'].fillna(db_master_df['U_Depot'])

    dispatch_delivery_df = pd.merge(
        left=dd_df, right=db_master_df[['CardCode', 'Final Depot to Bill']],
        left_on='CardCode', right_on='CardCode', how='left'
    )
    dispatch_delivery_df['Billing Status'] = np.where(
        dispatch_delivery_df['WhsCode'] == dispatch_delivery_df['Final Depot to Bill'], 'Matched', 'Mismatched'
    )

    date_columns = ['DocDate', 'LR/Dispatch Date', 'U_Delivery_Date']
    for col in date_columns:
        dispatch_delivery_df[col] = pd.to_datetime(dispatch_delivery_df[col], errors='coerce', dayfirst=True)

    current_date = pd.Timestamp('today').normalize()
    end_date_for_dispatch = dispatch_delivery_df['LR/Dispatch Date'].fillna(current_date)
    dispatch_delivery_df['Dispatch Days'] = (end_date_for_dispatch - dispatch_delivery_df['DocDate']).dt.days

    end_date_for_delivery = dispatch_delivery_df['U_Delivery_Date'].fillna(current_date)
    start_date_for_delivery = dispatch_delivery_df['LR/Dispatch Date'].fillna(dispatch_delivery_df['DocDate'])
    dispatch_delivery_df['Delivery Days'] = (end_date_for_delivery - start_date_for_delivery).dt.days


    # =================================================================================
    # --- SIDEBAR NAVIGATION ---
    # =================================================================================
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["📦 Stock Analysis", "🚚 Open Purchase Orders", "🧾 GRN vs. AP Reconciliation", "📈 Stock Aging", "🚚 Dispatch & Delivery"]
    )
    st.sidebar.divider()

    # =================================================================================
    # --- PAGE 1: Stock Analysis ---
    # =================================================================================
    if page == "📦 Stock Analysis":
        display_last_refreshed(['stockav', 'skulist', 'dsmmaster'], file_timestamps)
        st.sidebar.header("Stock Analysis Filters")
        
        state_group_options = sorted(stock_enriched['State_Group'].unique().tolist())
        selected_state_group = st.sidebar.multiselect("Filter by Region:", options=state_group_options, default=[])
        dsm_options = ["All"] + sorted(stock_enriched['RSM/ DSM'].unique().tolist())
        selected_dsm = st.sidebar.selectbox("Filter by DSM/RSM:", options=dsm_options)
        class_options = ["All"] + sorted(stock_enriched['Classification'].unique().tolist())
        selected_class = st.sidebar.selectbox("Filter by Classification:", options=class_options)
        if 'Prod Cat' in stock_enriched.columns:
            prod_cat_options = ["All"] + sorted(stock_enriched['Prod Cat'].unique().tolist())
            selected_prod_cat = st.sidebar.selectbox('Filter by Product Category:', options=prod_cat_options)
        remark_options = ["All"] + sorted(stock_enriched['Remark'].unique().tolist())
        selected_remark = st.sidebar.selectbox("Filter by TPU/TMD:", options=remark_options)
        
        st.header("Stock Analysis")
        filtered_stock_df = stock_enriched.copy()
        if selected_state_group:
           filtered_stock_df = filtered_stock_df[filtered_stock_df['State_Group'].isin(selected_state_group)]
        if selected_dsm != "All":
            filtered_stock_df = filtered_stock_df[filtered_stock_df['RSM/ DSM'] == selected_dsm]
        if selected_class != "All":
            filtered_stock_df = filtered_stock_df[filtered_stock_df['Classification'] == selected_class]
        if 'Prod Cat' in filtered_stock_df.columns and 'selected_prod_cat' in locals() and selected_prod_cat != "All":
            filtered_stock_df = filtered_stock_df[filtered_stock_df['Prod Cat'] == selected_prod_cat]
        if selected_remark != "All":
            filtered_stock_df = filtered_stock_df[filtered_stock_df['Remark'] == selected_remark]
        
        st.subheader("Key Stock Metrics")
        total_stock_value = filtered_stock_df['Stock Value'].sum()
        total_qty_cases = filtered_stock_df['Qty in Cases'].sum()
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Total Stock Value", format_indian_currency_kpi(total_stock_value))
        metric_col2.metric("Total Quantity in Cases", f"{total_qty_cases:,.0f}")
        
        st.subheader("Visual Analysis")
        view_selection = st.radio("Select View:", ("By Depot", "By Product Category", "By Vendor", "By DSM", "All Data"), horizontal=True)

        if not filtered_stock_df.empty:
            if view_selection in ["By Depot", "By DSM"]:
                try:
                    group_by_col = "WhsCode" if view_selection == "By Depot" else "RSM/ DSM"
                    chart_data = filtered_stock_df.groupby(group_by_col).agg({
                        'Stock Value': 'sum', 'Qty in Cases': 'sum'
                    }).reset_index()
                    chart_data = chart_data.sort_values(by='Stock Value', ascending=False)
                    chart_data['value_label'] = (chart_data['Stock Value'] / 1_00_000).apply(lambda x: f'₹{x:.2f}L')
                    chart_data['qty_label'] = chart_data['Qty in Cases']
                    title_text = f"Stock Value and Quantity {view_selection}"
                    fig = create_combo_chart(
                        df=chart_data, x_col=group_by_col, bar_col='Stock Value',
                        line_col='Qty in Cases', bar_text=chart_data['value_label'],
                        line_text=chart_data['qty_label'], title=title_text
                        )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create the chart. Error: {e}")

            st.subheader(f"Data View: {view_selection}")
            if view_selection == "All Data":
                csv_data = convert_df_to_csv(filtered_stock_df)
                st.download_button(label="Download Full Data as CSV", data=csv_data, file_name="stock_analysis_full_data.csv", mime="text/csv")
            else:
                sku_col = 'Item Description' if 'Item Description' in filtered_stock_df.columns else 'ItemCode'
                summary_cols_map = {
                    "By Depot": ['WhsCode', 'RSM/ DSM'],
                    "By Product Category": ['Prod Cat', 'Classification','WhsCode' ,sku_col],
                    "By Vendor": ['Vendor', sku_col],
                    "By DSM": ['RSM/ DSM', 'WhsCode']
                }
                summary_cols = summary_cols_map.get(view_selection)
                if summary_cols:
                    existing_summary_cols = [col for col in summary_cols if col in filtered_stock_df.columns]
                    if existing_summary_cols:
                        summary_df = filtered_stock_df.groupby(existing_summary_cols).agg({'Stock Value': 'sum', 'Qty in Cases': 'sum'}).reset_index()
                        st.dataframe(summary_df.style.format({'Stock Value': format_indian_currency, 'Qty in Cases': "{:,.0f}"}))
                        csv_data = convert_df_to_csv(summary_df)
                        st.download_button(
                            label=f"Download Summary ({view_selection}) as CSV",
                            data=csv_data,
                            file_name=f"stock_summary_{view_selection.replace(' ', '_').lower()}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"Could not create summary view for '{view_selection}' as required columns are missing.")
        else:
            st.warning("No data available for the selected filters.")


    # =================================================================================
    # --- PAGE 2: Open Purchase Orders ---
    # =================================================================================
    elif page == "🚚 Open Purchase Orders":
        display_last_refreshed(['openpo'], file_timestamps)
        st.sidebar.header("Open PO Filters")
        
        st.sidebar.subheader("Filter by Date")
        if 'DocDate' in open_po_data.columns and not open_po_data['DocDate'].dropna().empty:
            min_date, max_date = open_po_data['DocDate'].min().to_pydatetime().date(), open_po_data['DocDate'].max().to_pydatetime().date()
            from_date = st.sidebar.date_input("From Posting Date", value=min_date, min_value=min_date, max_value=max_date)
            to_date = st.sidebar.date_input("To Posting Date", value=max_date, min_value=min_date, max_value=max_date)
        else:
            from_date, to_date = None, None
            st.sidebar.info("Posting Date column not available for filtering.")
        
        st.sidebar.divider()
        st.sidebar.subheader("Other Filters")
        vendor_options = ["All"] + sorted(open_po_data['CardName'].unique().tolist())
        selected_vendor = st.sidebar.selectbox("Select a Vendor:", options=vendor_options)
        sku_options = ["All"] + sorted(open_po_data['Dscription'].unique().tolist())
        select_sku = st.sidebar.selectbox("Select a SKU:", options=sku_options)
        
        st.header("Pending Deliveries by Vendor (Open POs)")
        filtered_po_data = open_po_data.copy()
        if from_date and to_date:
            from_datetime, to_datetime = pd.to_datetime(from_date), pd.to_datetime(to_date)
            filtered_po_data = filtered_po_data[(filtered_po_data['DocDate'] >= from_datetime) & (filtered_po_data['DocDate'] <= to_datetime)]
        if selected_vendor != "All":
            filtered_po_data = filtered_po_data[filtered_po_data['CardName'] == selected_vendor]
        if select_sku != "All":
            filtered_po_data = filtered_po_data[filtered_po_data['Dscription'] == select_sku]
        
        st.subheader("Key Open PO Metrics")
        total_open_value = filtered_po_data['Open Order Value'].sum()
        total_open_qty = filtered_po_data['Qty in Cases'].sum()
        col1_po, col2_po = st.columns(2)
        col1_po.metric("Total Open Order Value", format_indian_currency_kpi(total_open_value))
        col2_po.metric("Total Open Quantity in Cases", f"{total_open_qty:,.0f}")
        
        st.subheader("Open Order Value by Vendor Name")
        if not filtered_po_data.empty:
            vendor_summary = filtered_po_data.groupby('CardName')['Open Order Value'].sum().reset_index()
            vendor_summary['Value (L)'] = vendor_summary['Open Order Value'] / 1_00_000
            vendor_summary['text_label'] = vendor_summary['Value (L)'].apply(lambda x: f'{x:.2f}L')
            fig_po = px.bar(vendor_summary, x='CardName', y='Value (L)', title='Total Open Order Value by Vendor', text='text_label', labels={'Value (L)': 'Total Value (in Lakhs)', 'CardName': 'Vendor'},color_discrete_sequence=['#9673FF'])
            fig_po.update_traces(textposition="outside")
            st.plotly_chart(fig_po, use_container_width=True)
            
            st.subheader("Open PO Aging Summary")
            if 'Open PO Aging (Days)' in filtered_po_data.columns and not filtered_po_data.empty:
                bins, labels = [-1, 10, 20, 30, 40, float('inf')], ['0-10 Days', '11-20 Days', '21-30 Days', '31-40 Days', '41+ Days']
                filtered_po_data['Aging Bucket'] = pd.cut(filtered_po_data['Open PO Aging (Days)'], bins=bins, labels=labels, right=True)
                aging_summary = filtered_po_data.groupby('Aging Bucket').agg(num_pos=('DocNum', 'nunique'), total_value=('Open Order Value', 'sum')).reset_index()
                aging_summary.rename(columns={'Aging Bucket': 'Aging Category', 'num_pos': 'Number of POs', 'total_value': 'Total Open Value'}, inplace=True)
                if not aging_summary.empty:
                    st.dataframe(aging_summary.style.format({'Total Open Value': format_indian_currency_kpi}), use_container_width=True, hide_index=True)
                    csv_aging_summary = convert_df_to_csv(aging_summary)
                    st.download_button(label="Download Aging Summary as CSV", data=csv_aging_summary, file_name="open_po_aging_summary.csv", mime="text/csv")
                else:
                    st.info("No aging data to display for the current selection.")

            csv_po_data = convert_df_to_csv(filtered_po_data)
            st.download_button(label="Download PO Details as CSV", data=csv_po_data, file_name="open_po_details.csv", mime="text/csv")
        else:
            st.info("No open purchase orders found for the selected filters.")

    # =================================================================================
    # --- PAGE 3: GRN vs. AP Reconciliation ---
    # =================================================================================
    elif page == "🧾 GRN vs. AP Reconciliation":
        display_last_refreshed(['grn', 'apbooking'], file_timestamps)
        st.sidebar.info("This page does not have any filters.")
        
        st.header("GRN vs. Accounts Payable Reconciliation")
        grn_docs, ap_grns = data['grn']['DocNum'].dropna().unique(), data['apbooking']['GRN No.'].dropna().unique()
        matched_grns, unmatched_grns = set(grn_docs).intersection(set(ap_grns)), set(grn_docs).difference(set(ap_grns))
        
        st.subheader("Reconciliation Summary")
        col1_grn, col2_grn, col3_grn = st.columns(3)
        col1_grn.metric("Total GRNs", len(grn_docs))
        col2_grn.metric("GRNs with AP Booking (Matched)", len(matched_grns))
        col3_grn.metric("GRNs without AP Booking (Unmatched)", len(unmatched_grns), delta_color="inverse")
        
        tab1, tab2 = st.tabs(["Unmatched GRNs (Pending AP Booking)", "Matched GRNs"])
        with tab1:
            st.subheader("Details of Unmatched GRNs")
            unmatched_df = data['grn'][data['grn']['DocNum'].isin(list(unmatched_grns))]
            st.dataframe(unmatched_df)
            csv_unmatched_grn = convert_df_to_csv(unmatched_df)
            st.download_button(label="Download Unmatched GRNs as CSV", data=csv_unmatched_grn, file_name="unmatched_grns.csv", mime="text/csv")
        with tab2:
            st.subheader("Details of Matched GRNs")
            matched_df = data['grn'][data['grn']['DocNum'].isin(list(matched_grns))]
            csv_matched_grn = convert_df_to_csv(matched_df)
            st.download_button(label="Download Matched GRNs as CSV", data=csv_matched_grn, file_name="matched_grns.csv", mime="text/csv")

    # =================================================================================
    # --- PAGE 4: Stock Aging ---
    # =================================================================================
    elif page == "📈 Stock Aging":
        display_last_refreshed(['stockaging'], file_timestamps)
        st.sidebar.header("Dashboard Filters")
        filtered_df = stock_aging_data.copy()

        brand_options = ['All'] + sorted(list(stock_aging_data['Brand'].unique()))
        selected_brand = st.sidebar.selectbox("Select Brand", brand_options)
        if selected_brand != 'All':
            filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]

        state_options = ['All'] + sorted(list(stock_aging_data['State'].unique()))
        selected_state = st.sidebar.selectbox("Select State", state_options)
        if selected_state != 'All':
            filtered_df = filtered_df[filtered_df['State'] == selected_state]

        warehouse_options = ['All'] + sorted(list(stock_aging_data['WhsCode'].unique()))
        selected_warehouse = st.sidebar.selectbox("Select Warehouse Code", warehouse_options)
        if selected_warehouse != 'All':
            filtered_df = filtered_df[filtered_df['WhsCode'] == selected_warehouse]

        prod_group_options = ['All'] + sorted(list(stock_aging_data['ProductGroup'].unique()))
        selected_prod_group = st.sidebar.selectbox("Select SKU", prod_group_options)
        if selected_prod_group != 'All':
            filtered_df = filtered_df[filtered_df['ProductGroup'] == selected_prod_group]

        st.title("📦 Interactive Stock Aging Dashboard")
        st.markdown("Use the filters on the left to analyze the inventory data.")

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            total_value = filtered_df["TotalValue"].sum()
            total_qty = filtered_df["TotalQty"].sum()
            aged_value = filtered_df[["181-360Value", "361-720Value", "721+DaysValue"]].sum().sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Inventory Value", format_indian_currency_kpi(total_value))
            col2.metric("Total Items in Stock", f"{total_qty:,.0f}")
            col3.metric("Value of Aged Stock (>180d)", format_indian_currency_kpi(aged_value))
            st.markdown("---")
            
            st.subheader("Stock by Age Category")
            less_than_30_value = filtered_df[['0-15Value', '16-30Value']].sum().sum()
            greater_than_30_value = total_value - less_than_30_value
            less_than_30_cases = filtered_df[['0-15Qty', '16-30Qty']].sum().sum()
            greater_than_30_cases = total_qty - less_than_30_cases

            kpi_col1, kpi_col2 = st.columns(2)
            kpi_col1.metric("Stock Value < 30 Days", format_indian_currency_kpi(less_than_30_value))
            kpi_col2.metric("Stock Value > 30 Days", format_indian_currency_kpi(greater_than_30_value))

            kpi_col3, kpi_col4 = st.columns(2)
            kpi_col3.metric("Stock Cases < 30 Days", format_indian_number(less_than_30_cases))
            kpi_col4.metric("Stock Cases > 30 Days", format_indian_number(greater_than_30_cases))
            st.markdown("---")
            
            st.header("Inventory Age Distribution")
            aging_data = {
                'Aging Bucket': ['0-15 Days', '16-30 Days', '31-60 Days', '61-90 Days', '91-180 Days', '181-360 Days', '361-720 Days', '721+ Days'],
                'Value': [
                    filtered_df['0-15Value'].sum(), filtered_df['16-30Value'].sum(),
                    filtered_df['31-60Value'].sum(), filtered_df['61-90Value'].sum(),
                    filtered_df['91-180Value'].sum(), filtered_df['181-360Value'].sum(),
                    filtered_df['361-720Value'].sum(), filtered_df['721+DaysValue'].sum()
                ],
                'Quantity': [
                    filtered_df['0-15Qty'].sum(), filtered_df['16-30Qty'].sum(),
                    filtered_df['31-60Qty'].sum(), filtered_df['61-90Qty'].sum(),
                    filtered_df['91-180Qty'].sum(), filtered_df['181-360Qty'].sum(),
                    filtered_df['361-720Qty'].sum(), filtered_df['721+Qty'].sum()
                ]
            }
            aging_df = pd.DataFrame(aging_data)
            aging_df['value_label'] = (aging_df['Value'] / 1_00_000).apply(lambda x: f'₹{x:.2f}L')
            aging_df['qty_label'] = aging_df['Quantity'].apply(format_indian_number)

            fig_aging_combo = create_combo_chart(
                df=aging_df, x_col='Aging Bucket', bar_col='Value', line_col='Quantity',
                bar_text=aging_df['value_label'], line_text=aging_df['qty_label'],
                title="Inventory Age Distribution: Value (Bars) and Quantity (Line)"
            )
            st.plotly_chart(fig_aging_combo, use_container_width=True)
            st.markdown("---")
            
            st.header("Brand Value by Age")
            filtered_df['<30 Days Value'] = filtered_df['0-15Value'] + filtered_df['16-30Value']
            filtered_df['>30 Days Value'] = filtered_df['TotalValue'] - (filtered_df['0-15Value'] + filtered_df['16-30Value'])
            
            lt30_by_brand = filtered_df[filtered_df['<30 Days Value'] > 0].groupby('Brand')['<30 Days Value'].sum().reset_index()
            gt30_by_brand = filtered_df[filtered_df['>30 Days Value'] > 0].groupby('Brand')['>30 Days Value'].sum().reset_index()

            col1_brand, col2_brand = st.columns(2)
            with col1_brand:
                if not lt30_by_brand.empty:
                    lt30_by_brand_chart = lt30_by_brand.sort_values('<30 Days Value', ascending=False)
                    lt30_by_brand_chart['Value (L)'] = lt30_by_brand_chart['<30 Days Value'] / 1_00_000
                    lt30_by_brand_chart['text_label'] = lt30_by_brand_chart['Value (L)'].apply(lambda x: f'{x:.2f}L')
                    fig_lt30 = px.bar(
                        lt30_by_brand_chart, x='Brand', y='Value (L)',
                        title='Stock Value (< 30 Days) by Brand', text='text_label',
                        labels={'Value (L)': 'Value (in Lakhs)'}
                    )
                    fig_lt30.update_traces(textposition='outside')
                    st.plotly_chart(fig_lt30, use_container_width=True)
                else:
                    st.info("No stock data available for < 30 Days.")
            with col2_brand:
                if not gt30_by_brand.empty:
                    gt30_by_brand_chart = gt30_by_brand.sort_values('>30 Days Value', ascending=False)
                    gt30_by_brand_chart['Value (L)'] = gt30_by_brand_chart['>30 Days Value'] / 1_00_000
                    gt30_by_brand_chart['text_label'] = gt30_by_brand_chart['Value (L)'].apply(lambda x: f'{x:.2f}L')
                    fig_gt30 = px.bar(
                        gt30_by_brand_chart, x='Brand', y='Value (L)',
                        title='Stock Value (> 30 Days) by Brand', text='text_label',
                        labels={'Value (L)': 'Value (in Lakhs)'}, color_discrete_sequence=['#d62728']
                    )
                    fig_gt30.update_traces(textposition='outside')
                    st.plotly_chart(fig_gt30, use_container_width=True)
                else:
                    st.info("No stock data available for > 30 Days.")
            
            with st.expander("View Brand Value by Age Data Tables"):
                table_col1, table_col2 = st.columns(2)
                with table_col1:
                    st.write("**Stock Value < 30 Days**")
                    if not lt30_by_brand.empty:
                        lt30_display = lt30_by_brand.rename(columns={'<30 Days Value': 'Total Value'})
                        lt30_display = lt30_display.sort_values('Total Value', ascending=False)
                        st.dataframe(
                            lt30_display.style.format({'Total Value': format_indian_currency}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No data available.")
                with table_col2:
                    st.write("**Stock Value > 30 Days**")
                    if not gt30_by_brand.empty:
                        gt30_display = gt30_by_brand.rename(columns={'>30 Days Value': 'Total Value'})
                        gt30_display = gt30_display.sort_values('Total Value', ascending=False)
                        st.dataframe(
                            gt30_display.style.format({'Total Value': format_indian_currency}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No data available.")
            
            st.header("Download Filtered Stock Data")
            csv_aging_data = convert_df_to_csv(filtered_df)
            st.download_button(label="Download Aging Data as CSV", data=csv_aging_data, file_name="stock_aging_data.csv", mime="text/csv")

    # =================================================================================
    # --- PAGE 5: Dispatch & Delivery ---
    # =================================================================================
    elif page == "🚚 Dispatch & Delivery":
        display_last_refreshed(['dbmaster', 'dd', 'depot_mapping'], file_timestamps)
        st.header("Billing and Logistics Performance")
        
        st.sidebar.header("Logistics Filters")
        st.sidebar.subheader("Filter by Date")
        if 'DocDate' in dispatch_delivery_df.columns and not dispatch_delivery_df['DocDate'].dropna().empty:
            min_date_dd, max_date_dd = dispatch_delivery_df['DocDate'].min().to_pydatetime().date(), dispatch_delivery_df['DocDate'].max().to_pydatetime().date()
            from_date_dd = st.sidebar.date_input("From Posting Date", value=min_date_dd, min_value=min_date_dd, max_value=max_date_dd, key="dd_from_date")
            to_date_dd = st.sidebar.date_input("To Posting Date", value=max_date_dd, min_value=min_date_dd, max_value=max_date_dd, key="dd_to_date")
        else:
            from_date_dd, to_date_dd = None, None
            st.sidebar.info("Posting Date column not available for filtering.")
        
        st.sidebar.divider()
        st.sidebar.subheader("Other Filters")
        if from_date_dd and to_date_dd:
            from_datetime_dd, to_datetime_dd = pd.to_datetime(from_date_dd), pd.to_datetime(to_date_dd)
            date_filtered_df = dispatch_delivery_df[(dispatch_delivery_df['DocDate'] >= from_datetime_dd) & (dispatch_delivery_df['DocDate'] <= to_datetime_dd)]
        else:
            date_filtered_df = dispatch_delivery_df.copy()
        
        unique_depots = sorted(date_filtered_df['WhsCode'].dropna().unique())
        depot_selection = st.sidebar.multiselect('Select Depot(s) to Analyze:', options=unique_depots, default=[])
        filtered_df = date_filtered_df[date_filtered_df['WhsCode'].isin(depot_selection)] if depot_selection else date_filtered_df.copy()
        
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            st.subheader("Billing Accuracy")
            matched_count = filtered_df[filtered_df['Billing Status'] == 'Matched'].shape[0]
            mismatched_count = filtered_df[filtered_df['Billing Status'] == 'Mismatched'].shape[0]
            total_orders_kpi = matched_count + mismatched_count
            
            if total_orders_kpi > 0:
                matched_percentage = f"{(matched_count / total_orders_kpi) * 100:.1f}%"
                mismatched_percentage = f"{(mismatched_count / total_orders_kpi) * 100:.1f}%"
            else:
                matched_percentage, mismatched_percentage = "N/A", "N/A"
            
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("✅ Correctly Billed Orders", f"{matched_count:,}", delta=matched_percentage)
            kpi2.metric("❌ Cross Billed Orders", f"{mismatched_count:,}", delta=mismatched_percentage, delta_color="inverse")

            def categorize_days(days):
                if pd.isna(days) or days < 0: return 'Invalid Data'
                elif days <= 1: return '0-1 Day'
                elif days == 2: return '2 Days'
                elif days == 3: return '3 Days'
                elif days <= 5: return '4-5 Days'
                else: return 'More than 5 Days'

            filtered_df['Dispatch Category'] = filtered_df['Dispatch Days'].apply(categorize_days)
            filtered_df['Delivery Category'] = filtered_df['Delivery Days'].apply(categorize_days)

            st.subheader("Dispatch & Delivery Timelines")
            dispatch_le3 = filtered_df[filtered_df['Dispatch Days'] <= 3].shape[0]
            dispatch_gt3 = filtered_df[filtered_df['Dispatch Days'] > 3].shape[0]
            delivery_le3 = filtered_df[filtered_df['Delivery Days'] <= 3].shape[0]
            delivery_gt3 = filtered_df[filtered_df['Delivery Days'] > 3].shape[0]

            if total_orders_kpi > 0:
                dispatch_le3_pct = f"{(dispatch_le3 / total_orders_kpi) * 100:.1f}%"
                dispatch_gt3_pct = f"{(dispatch_gt3 / total_orders_kpi) * 100:.1f}%"
                delivery_le3_pct = f"{(delivery_le3 / total_orders_kpi) * 100:.1f}%"
                delivery_gt3_pct = f"{(delivery_gt3 / total_orders_kpi) * 100:.1f}%"
            else:
                dispatch_le3_pct, dispatch_gt3_pct, delivery_le3_pct, delivery_gt3_pct = "N/A", "N/A", "N/A", "N/A"

            kpi3, kpi4, kpi5, kpi6 = st.columns(4)
            kpi3.metric("🚚 Dispatched <= 3 Days", f"{dispatch_le3:,}", delta=dispatch_le3_pct)
            kpi4.metric("⏳ Dispatched > 3 Days", f"{dispatch_gt3:,}", delta=dispatch_gt3_pct, delta_color="inverse")
            kpi5.metric("📦 Delivered <= 3 Days", f"{delivery_le3:,}", delta=delivery_le3_pct)
            kpi6.metric("🐌 Delivered > 3 Days", f"{delivery_gt3:,}", delta=delivery_gt3_pct, delta_color="inverse")
            st.divider()

            st.subheader("Dispatch vs. Delivery Performance")
            col1_perf, col2_perf = st.columns(2)
            with col1_perf:
                matched_df = filtered_df[filtered_df['Billing Status'] == 'Matched']
                if not matched_df.empty:
                    st.altair_chart(create_grouped_bar_chart(matched_df, '✅ Matched Orders'), use_container_width=True)
                else:
                    st.info("No Matched orders data to display.")

            with col2_perf:
                mismatched_df = filtered_df[filtered_df['Billing Status'] == 'Mismatched']
                if not mismatched_df.empty:
                    st.altair_chart(create_grouped_bar_chart(mismatched_df, '❌ Cross Billed Orders'), use_container_width=True)
                else:
                    st.info("No Cross Billed orders data to display.")

            st.divider()
            st.subheader("Depot Performance Matrix By Orders")
            depot_summary = dispatch_delivery_df.groupby('WhsCode').agg(
                total_orders=('WhsCode', 'size'),
                matched_orders=('Billing Status', lambda x: (x == 'Matched').sum()),
                mismatched_orders=('Billing Status', lambda x: (x == 'Mismatched').sum())
            ).reset_index()
            depot_summary['match_rate_%'] = (depot_summary['matched_orders'] / depot_summary['total_orders']) * 100
            depot_summary['mismatch_rate_%'] = (depot_summary['mismatched_orders'] / depot_summary['total_orders']) * 100
            depot_summary = depot_summary.rename(columns={'WhsCode': 'Depot', 'total_orders': 'Total Orders', 'matched_orders': 'Matched', 'mismatched_orders': 'Cross Billed', 'match_rate_%': 'Match Rate', 'mismatch_rate_%': 'Cross Billed Rate'})
            depot_summary = depot_summary.sort_values(by='Match Rate', ascending=False)
            st.dataframe(depot_summary.style.format({'Match Rate': '{:.2f}%', 'Cross Billed Rate': '{:.2f}%'}), use_container_width=True, hide_index=True)

            st.subheader("Depot Performance Matrix By Cases")
            if 'Qty in Cases' in dispatch_delivery_df.columns and dispatch_delivery_df['Qty in Cases'].sum() > 0:
                depot_summary_cases_raw = dispatch_delivery_df.groupby(['WhsCode', 'Billing Status'])['Qty in Cases'].sum().unstack(fill_value=0)
                if 'Matched' not in depot_summary_cases_raw.columns: depot_summary_cases_raw['Matched'] = 0
                if 'Mismatched' not in depot_summary_cases_raw.columns: depot_summary_cases_raw['Mismatched'] = 0
                depot_summary_cases = depot_summary_cases_raw.reset_index()
                depot_summary_cases['total_cases'] = depot_summary_cases['Matched'] + depot_summary_cases['Mismatched']
                depot_summary_cases['match_rate_%'] = (depot_summary_cases['Matched'] / depot_summary_cases['total_cases'] * 100).fillna(0)
                depot_summary_cases['mismatch_rate_%'] = (depot_summary_cases['Mismatched'] / depot_summary_cases['total_cases'] * 100).fillna(0)
                depot_summary_cases = depot_summary_cases.rename(columns={'WhsCode': 'Depot', 'total_cases': 'Total Cases', 'Matched': 'Matched Cases', 'Mismatched': 'Cross Billed Cases', 'match_rate_%': 'Match Rate', 'mismatch_rate_%': 'Cross Billed Rate'})
                depot_summary_cases = depot_summary_cases.sort_values(by='Match Rate', ascending=False)
                depot_summary_cases = depot_summary_cases[['Depot', 'Total Cases', 'Matched Cases', 'Cross Billed Cases', 'Match Rate', 'Cross Billed Rate']]
                st.dataframe(depot_summary_cases.style.format({'Total Cases': '{:,.0f}', 'Matched Cases': '{:,.0f}', 'Cross Billed Cases': '{:,.0f}', 'Match Rate': '{:.2f}%', 'Cross Billed Rate': '{:.2f}%'}), use_container_width=True, hide_index=True)
            else:
                st.info("No case quantity data available for the case-based performance matrix.")

            st.subheader("Download Detailed Logistics Data")
            csv_log_data = convert_df_to_csv(filtered_df)
            st.download_button(label="Download Filtered Logistics Data as CSV", data=csv_log_data, file_name="dispatch_delivery_data.csv", mime="text/csv")

else:
    st.error("Failed to load data from the FTP server. The dashboard cannot be displayed.")
    st.warning("Please check the FTP connection details in your Streamlit secrets and ensure the server is accessible.")