import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# def format_currency(value):
#     return f"$ {value:,.2f}"


def process_loan_data(input_file, output_file):
    """ This function processes the data provided by DV01 and saves the output """
    # Load the data
    df = pd.read_csv(input_file, skiprows=1)

    # Step 1: Filter relevant columns from the input files provided by DV01
    relevant_columns = ['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'term', 'issue_d', 'grade', 'loan_status',
                        'total_pymnt', 'total_rec_prncp', 'total_rec_int']
    data = df[relevant_columns]
    data.dropna(subset=['int_rate'], inplace=True)

    # Step 2: Data Cleaning
    # Rename columns for clarity and also match the Output file provided
    data.rename(columns={
        'loan_amnt': 'LoanAmount',
        'funded_amnt': 'FundedAmount',
        'int_rate': 'InterestRate',
        'term': 'Term',
        'issue_d': 'IssueDate',
        'grade': 'Grade',
        'loan_status': 'LoanStatus',
        'total_pymnt': 'TotalPayments',
        'total_rec_prncp': 'PrincipalReceived',
        'total_rec_int': 'InterestReceived'
    }, inplace=True)

    # Convert 'InterestRate' from string to numeric
    data['InterestRate'] = data['InterestRate'].str.replace('%', '', regex=True).str.strip()
    data['InterestRate'] = pd.to_numeric(data['InterestRate'], errors='coerce')

    # Convert 'InterestRate' to a percentage if stored as a fraction
    data['InterestRate'] = data['InterestRate'] * 100 if data['InterestRate'].max() <= 1 else data['InterestRate']

    # Convert 'IssueDate' to datetime
    data['IssueDate'] = pd.to_datetime(data['IssueDate'], errors='coerce')

    # Combine F and G grades into F-G by summing all F and G graded Issues
    data['Grade'] = data['Grade'].apply(lambda x: 'F-G' if x in ['F', 'G'] else x)

    # Step 3: Add Derived Columns
    # Categorize loans by status
    status_mapping = {
        'Fully Paid': 'Fully Paid',
        'Current': 'Current',
        'Late (31-120 days)': 'Late',
        'Charged Off': 'Charged Off'
    }
    data['LoanCategory'] = data['LoanStatus'].map(status_mapping)

    # Step 4: Aggregation
    # Define the weighted average function
    def weighted_avg(df, value_col, weight_col):
        return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum() if df[weight_col].sum() > 0 else 0

    # Aggregate metrics by Grade
    summary = data.groupby('Grade').apply(
        lambda x: pd.Series({
            'Total Issued': x['LoanAmount'].sum(),
            'Fully Paid': x.loc[x['LoanCategory'] == 'Fully Paid', 'LoanAmount'].sum(),
            'Current': x.loc[x['LoanCategory'] == 'Current', 'LoanAmount'].sum(),
            'Late': x.loc[x['LoanCategory'] == 'Late', 'LoanAmount'].sum(),
            'Charged Off': x.loc[x['LoanCategory'] == 'Charged Off', 'LoanAmount'].sum(),
            'Principal Payments Received': x['PrincipalReceived'].sum(),
            'Interest Payments Received': x['InterestReceived'].sum(),
            'Avg Interest Rate': weighted_avg(x, 'InterestRate', 'LoanAmount')
        })
    ).reset_index()


    # Calculate total average interest rate before string conversion
    # The Total AVG Interest Rate is calculated as the weighted average using the TotalIssued amount for each grade  * the AVG Interest Rate / total_amount
    total_avg_rate = (summary['Total Issued'] * summary['Avg Interest Rate']).sum() / summary['Total Issued'].sum()
    
    # Add a Total row
    total_row = pd.Series({
        'Grade': 'Total',
        'Total Issued': summary['Total Issued'].sum(),
        'Fully Paid': summary['Fully Paid'].sum(),
        'Current': summary['Current'].sum(),
        'Late': summary['Late'].sum(),
        'Charged Off': summary['Charged Off'].sum(),
        'Principal Payments Received': summary['Principal Payments Received'].sum(),
        'Interest Payments Received': summary['Interest Payments Received'].sum(),
        'Avg Interest Rate': total_avg_rate
    })
    
    # Format currency columns to reflect the provided output
    # currency_columns = ['Total Issued', 'Fully Paid', 'Current', 'Late', 'Charged Off', 
    #                    'Principal Payments Received', 'Interest Payments Received']
    
    # for col in currency_columns:
    #     summary[col] = summary[col].apply(format_currency)
    #     total_row[col] = format_currency(total_row[col])
    
    # Round numeric values before string conversion
    summary['Avg Interest Rate'] = summary['Avg Interest Rate'].round(2)
    total_row['Avg Interest Rate'] = round(total_row['Avg Interest Rate'], 2)
    
    # Add percentage symbol after rounding to reflect the output file provided
    summary['Avg Interest Rate'] = summary['Avg Interest Rate'].astype(str) + '%'
    total_row['Avg Interest Rate'] = str(total_row['Avg Interest Rate']) + '%'
    
    # Combine summary with total row
    summary = pd.concat([summary, total_row.to_frame().T], ignore_index=True)

    # Step 5: Export Results
    summary.to_excel(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

    return summary  # Return the summary DataFrame

# Process each dataset
files = {
    "2018Q3": "Data/LoanStats_securev1_2018Q3.csv",
    "2018Q4": "Data/LoanStats_securev1_2018Q4.csv",
    "2019Q1": "Data/LoanStats_securev1_2019Q1.csv"
}

# Saving the output files and generating charts
output_directory = "Output/"
quarterly_data = {}

for quarter, file_path in files.items():
    output_path = f"{output_directory}{quarter}.xlsx"
    summary = process_loan_data(file_path, output_path)
    quarterly_data[quarter] = summary


""" 
Creating Charts to visualize the Charge Off, Oustanding Balance, and the AVG Interest Rates for all Quarters for all Grades

"""
# Chart 1: Avg Interest Rate for All Quarters (Line chart)
avg_interest_rate_data = []
for quarter, summary in quarterly_data.items():
    avg_interest_rate_data.append(
        {
            'Grade': summary['Grade'],
            'Avg Interest Rate': summary['Avg Interest Rate'].str.replace('%', '').astype(float),
            'Quarter': quarter
        }
    )
avg_interest_rate_df = pd.concat([pd.DataFrame(data) for data in avg_interest_rate_data])

plt.figure(figsize=(10, 6))
sns.lineplot(
    x='Grade',
    y='Avg Interest Rate',
    hue='Quarter',
    data=avg_interest_rate_df,
    marker='o'
)
plt.title("Average Interest Rate by Grade for All Quarters")
plt.ylabel("Average Interest Rate (%)")
plt.xticks(rotation=45)
plt.show()

# Chart 2: Current Loan Amounts for All Quarters (Bar Chart)
current_data = []
for quarter, summary in quarterly_data.items():
    current_data.append(
        {
            'Grade': summary['Grade'],
            'CurrentLoanAmount': summary['Current'],
            'Quarter': quarter
        }
    )

current_df = pd.concat([pd.DataFrame(data) for data in current_data])

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Grade',
    y='CurrentLoanAmount',
    hue='Quarter',
    data=current_df
)
plt.title('Current Loan Amounts by Grade for All Quarters')
plt.ylabel('Loan Amount ($)')
plt.xticks(rotation=45)
plt.show()

# Create Chart 3: Charged Off Loan Amounts for All Quarters (Bar Chart)
charged_off_data = []
for quarter, summary in quarterly_data.items():
    charged_off_data.append(
        {
            'Grade': summary['Grade'],
            'ChargedOffLoanAmount': summary['Charged Off'],
            'Quarter': quarter
        }
    )

charged_off_df = pd.concat([pd.DataFrame(data) for data in charged_off_data])

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Grade',
    y='ChargedOffLoanAmount',
    hue='Quarter',
    data=charged_off_df
)
plt.title('Charged Off Loan Amounts by Grade for All Quarters')
plt.ylabel('Loan Amount ($)')
plt.xticks(rotation=45)
plt.show()