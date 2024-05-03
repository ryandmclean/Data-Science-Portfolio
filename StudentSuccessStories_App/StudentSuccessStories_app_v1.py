# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# Get the current credentials
session = get_active_session()

#################################################
# Defining the Filters and Selectors for the App
#################################################

slider_fontsize = st.slider(
    "Select a Font Size",
    min_value=8,
    max_value=20,
    value=12,
    help="Enter the main font size",
)

# Function to fetch distinct months from the database
def fetch_distinct_months():
    query = "SELECT DISTINCT MONTH FROM mma_dw.dm.mma_user_monthly_growth_report ORDER BY MONTH DESC"
    # Assuming session is your Snowflake or database session, adjust as necessary
    result = session.sql(query).to_pandas()
    months = result['MONTH'].unique().tolist()
    return months

# Dropdown for selecting the month, defaulting to the most recent month
selected_month = st.selectbox(
    'Select a Month',
    fetch_distinct_months(),
    index=0,
    format_func=lambda x: pd.to_datetime(x).strftime('%B %Y')  # Format for readability
)

# Dropdown for selecting the organization name
selected_organization = st.selectbox(
    'Select an Organization Name',
    ['Clark County School District', 'Palm Beach ELC']
)

# Dropdown for selecting the academy
selected_academy = st.selectbox(
    'Select a Product',
    ['My Math Academy', 'My Reading Academy']
)

# Map the selected academy to the corresponding database table
academy_to_table_map = {
    'My Math Academy': 'mma_dw.dm.mma_user_monthly_growth_report',
    'My Reading Academy': 'mra_dw.dm.mra_user_monthly_growth_report'
}
table_name = academy_to_table_map[selected_academy]

slider_total_hours = st.slider(
    "Minimum Usage Hours",
    min_value=0,
    max_value=40,
    value=5,
    help="Enter the minimum usage hours for the student",
)

slider_mins_week = st.slider(
    "Minimum Minutes per Week",
    min_value=1,
    max_value=60,
    value=20,
    help="Enter the minimum minutes per week for the students",
)

slider_mins_week_max = st.slider(
    "Maximum Minutes per Week",
    min_value=45,
    max_value=75,
    value=60,
    help="Enter the maximum minutes per week that you want to show",
)

bool_above_target = st.radio(
    "Only include students whose current NGR is above the progress target",
    options=[True, False],  # Options are True and False
    format_func=lambda x: str(x),  # Format the display to show 'True' or 'False' as strings
    index=0  # Set default selection to True (the first option in the list)
)



###################################
# Getting and Cleaning Data
###################################
# Use the selected values to construct the query
query = f"""
select * from {table_name} 
WHERE MONTH = '{selected_month}' 
AND ORGANIZATION_NAME = '{selected_organization}' 
AND IS_LICENSED_STUDENT = 1
AND HAS_PLACEMENT = 1
AND IS_CUMULATIVE_ACTIVE_STUDENT = 1
"""
# AND CURRENT_GRADE_READINESS >= MAX_GRADE_READINESS 

query_firstmonth = f"""
WITH RankedUsage AS (
    SELECT STUDENT_ID,ORGANIZATION_ID,SCHOOL_ID,CLASSROOM_ID,
    MONTH AS FIRST_MONTH, 
    CURRENT_GRADE_READINESS AS FIRST_CURRENT_GRADE_READINESS,
    MAX_GRADE_READINESS AS FIRST_MAX_GRADE_READINESS,
           ROW_NUMBER() OVER (PARTITION BY STUDENT_ID,ORGANIZATION_ID,SCHOOL_ID,CLASSROOM_ID ORDER BY MONTH ASC) AS rn
    FROM {table_name} 
    WHERE ORGANIZATION_NAME = '{selected_organization}' 
    AND BYPASSED_SKILL_CNT > 0
    AND IS_LICENSED_STUDENT = 1
    AND HAS_PLACEMENT = 1
    AND IS_CUMULATIVE_ACTIVE_STUDENT = 1
)
SELECT *
FROM RankedUsage
WHERE rn = 1;
"""


@st.cache_data
def get_data(query):
    return session.sql(query).to_pandas()


### Get the main monthly growth and apply filters
monthly_growth_data = get_data(query)
st.write("Total Students for district")
st.write(monthly_growth_data.groupby('MONTH')['STUDENT_ID'].nunique().reset_index(name='Unique_Student_Count'))
monthly_growth_data_filter = (
    ((monthly_growth_data['CURRENT_GRADE_READINESS'] >= monthly_growth_data['MAX_GRADE_READINESS']) if bool_above_target else True) &
    (monthly_growth_data['WEEKLY_USAGE_MIN'] >= slider_mins_week) & 
    (monthly_growth_data['WEEKLY_USAGE_MIN'] <= slider_mins_week_max) &
    (monthly_growth_data['TOTAL_PROGRAM_USAGE'] >= slider_total_hours * 3600)
)
monthly_growth_data = monthly_growth_data[monthly_growth_data_filter]
st.write("Total Students with sufficient usage")
st.write(monthly_growth_data.groupby('MONTH')['STUDENT_ID'].nunique().reset_index(name='Unique_Student_Count'))


# Additional preparation based on your existing setup
monthly_growth_data['TOTAL_BYPASSED_SKILL_CNT'] = monthly_growth_data['BYPASSED_SKILL_CNT'] + monthly_growth_data['PRIOR_BYPASSED_SKILL_CNT']
grades_order = ['Kindergarten', '1', '2']
monthly_growth_data['STUDENT_GRADE'] = pd.Categorical(monthly_growth_data['STUDENT_GRADE'], categories=grades_order, ordered=True)
monthly_growth_data['ACTIVE_WEEKS'] = monthly_growth_data['TOTAL_PROGRAM_USAGE'] / 60 / monthly_growth_data['WEEKLY_USAGE_MIN']

first_month_growth_data = get_data(query_firstmonth)
# st.subheader("Monthly Growth Data")
# st.dataframe(first_month_growth_data.head(100), use_container_width=True)

vars_to_keep = ['STUDENT_ID','ORGANIZATION_ID','SCHOOL_ID','CLASSROOM_ID','STUDENT_GRADE','ACTIVE_WEEKS','MONTH','CURRENT_GRADE_READINESS','MAX_GRADE_READINESS']
df_first_last = pd.merge(left=monthly_growth_data[vars_to_keep], 
                         right=first_month_growth_data, 
                         how='left', 
                         left_on=['STUDENT_ID','ORGANIZATION_ID','SCHOOL_ID','CLASSROOM_ID'],
                         right_on=['STUDENT_ID','ORGANIZATION_ID','SCHOOL_ID','CLASSROOM_ID'])
# st.subheader("Monthly Growth Data")
# st.dataframe(df_first_last.head(100), use_container_width=True)
st.write("Month of Placement for sample")
st.write(df_first_last.groupby('FIRST_MONTH')['STUDENT_ID'].nunique().reset_index(name='Unique_Student_Count'))


###################################
# Percentage of Kids on Track across time
###################################

# Step 1: Calculate statuses
def calculate_status(current, max):
    if current > max + 0.05:
        return 'Ahead'
    elif current > max - 0.05:
        return 'On Track'
    else:
        return 'Not on Track'

df_first_last['Status'] = df_first_last.apply(lambda row: calculate_status(row['CURRENT_GRADE_READINESS'], row['MAX_GRADE_READINESS']), axis=1)
df_first_last['First_Status'] = df_first_last.apply(lambda row: calculate_status(row['FIRST_CURRENT_GRADE_READINESS'], row['FIRST_MAX_GRADE_READINESS']), axis=1)

# Count the number of rows for each status for both first and current
status_counts = df_first_last['Status'].value_counts().reindex(['Not on Track', 'On Track', 'Ahead'], fill_value=0)
first_status_counts = df_first_last['First_Status'].value_counts().reindex(['Not on Track', 'On Track', 'Ahead'], fill_value=0)

# Prepare data for the stacked bar chart
categories = ['At Placement', 'Current Next Grade Readines']
data = np.array([first_status_counts, status_counts])

# Define the bottom for the stacked bars
bottom = np.zeros(2)

# Colors for each segment
colors = ['red', 'blue', 'green']
labels = ['Not on Track', 'On Track', 'Ahead']

# Calculate total counts for normalization
total_counts = data.sum(axis=1)

fig, ax = plt.subplots()

bottom = np.zeros(len(categories))

for i, (label, color) in enumerate(zip(labels, colors)):
    bars = ax.bar(categories, data[:, i], bottom=bottom, label=label, color=color)
    bottom += data[:, i]

    # Calculate percentage of each segment and add data labels
    for bar, count in zip(bars, data[:, i]):
        percentage_value = count / total_counts[i % len(categories)] * 100  # Calculate as float for comparison
        if percentage_value >= 5:  # Only display label if percentage is 5% or higher
            percentage_text = f'{percentage_value:.1f}%'  # Format percentage as string for display
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, percentage_text,
                    ha='center', va='center', color='white', fontsize=slider_fontsize)

# Adjust y-axis limits to create space for the legend above the bars
current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)  # Increase y-limit by 15%
ax.set_ylabel('')  # Remove y-axis label and tick marks
ax.set_yticks([])

# Adding details and adjusting legend position
legend_elements = [
    Patch(facecolor='red', label='Not on Track'),
    Patch(facecolor='blue', label='On Track'),
    Patch(facecolor='green', label='Ahead')
]
ax.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=slider_fontsize*0.75)

st.pyplot(fig)


###################################
# NGR Stacked Bar Percentages
###################################
# Calculate averages and percentages
avg_data = monthly_growth_data.groupby('STUDENT_GRADE')[['TOTAL_BYPASSED_SKILL_CNT', 'TOTAL_MASTERED_SKILL_CNT', 'REMAIN_SKILLS_CNT']].mean()
percent_data = avg_data.div(avg_data.sum(axis=1), axis=0) * 100

# Calculate the average MAX_GRADE_READINESS for each grade, converted to percentage
avg_max_readiness = (monthly_growth_data.groupby('STUDENT_GRADE')['MAX_GRADE_READINESS'].mean() * 100).to_dict()

# Plot horizontal stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_containers = percent_data.plot(kind='barh', stacked=True, color=['blue', 'green', 'red'], ax=ax, width=0.4)  # Adjust bar width as necessary

# Define a custom label formatting function
# def custom_label_formatter(x):
#     return f'{x:.0f}%'

# Manually add data labels for each color segment within the bars if the value is greater than 5%
for container in bar_containers.containers:
    for bar in container:
        # Calculate the label, which is the width of the bar segment
        width = bar.get_width()
        if width > 5:  # Only label the bar if its width is greater than 5%
            label_x_pos = bar.get_x() + width / 2
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.0f}%', 
                    ha='center', va='center', color='white', fontsize=slider_fontsize)

# Calculate and draw the progress target lines and annotations
for index, (grade, readiness) in enumerate(avg_max_readiness.items()):
    readiness_percentage = readiness  # Assuming readiness is already a percentage
    # Adjust these values to change the positioning relative to the bar
    line_y_start = index + 0.4 / 2  # Slightly above the bottom of the bar
    line_y_end = index - 0.4 / 2  # Slightly below the top of the bar

    # Draw a short vertical line for the progress target
    ax.plot([readiness_percentage, readiness_percentage], [line_y_start, line_y_end], color='black', linestyle='--')

    # Annotate below the line
    ax.annotate(f'Progress Target: {readiness_percentage:.0f}%', 
                xy=(readiness_percentage, line_y_start), 
                xytext=(readiness_percentage, line_y_start + 0.15),  # Text position below the line
                textcoords='data', fontsize=slider_fontsize, ha='center', 
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
                va='top')

# Customize the x-axis and legend as before
ax.set_ylabel("Student Grade",fontsize = slider_fontsize)
ax.set_xlabel('% Next Grade Readiness',fontsize = slider_fontsize)
ax.tick_params(axis='x', labelsize=slider_fontsize)
ax.tick_params(axis='y', labelsize=slider_fontsize)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_xticks(range(0, 101, 10))
ax.set_xlim([0, 100])
ax.invert_yaxis()
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin + 0.2, ymax) 

legend_elements = [
    Patch(facecolor='blue', label='Prior Knowledge'),
    Patch(facecolor='green', label='Completed'),
    Patch(facecolor='red', label='Remaining Skills')
]
ax.legend(handles=legend_elements, loc='upper center', ncol=3, title="", bbox_to_anchor=(0.5, 1.10),fontsize = slider_fontsize)

fig.suptitle('', fontsize=slider_fontsize, y=1.02)

# Display the plot in Streamlit
st.pyplot(fig)

###################################
# Comparing Beginning and Current NGR and Skills
###################################
# Copy and filter df_first_last as before
df_plotting = df_first_last.dropna(subset=['FIRST_MONTH']).copy()
df_plotting['FIRST_MONTH'] = pd.to_datetime(df_plotting['FIRST_MONTH'])
df_plotting.sort_values('FIRST_MONTH', inplace=True)
unique_months = df_plotting['FIRST_MONTH'].unique()

n = len(unique_months)
ncols = 2  # Reduce the number of columns to 2
nrows = n // ncols + (n % ncols > 0)

# Increase the figure size accordingly
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 6*nrows), constrained_layout=True)

for i, month in enumerate(unique_months):
    ax = axs.flatten()[i]
    df_filtered = df_plotting[df_plotting['FIRST_MONTH'] == month]
    averages = df_filtered[['CURRENT_GRADE_READINESS', 'MAX_GRADE_READINESS', 
                            'FIRST_CURRENT_GRADE_READINESS', 'FIRST_MAX_GRADE_READINESS']].mean()

    x_coords = np.array([0, 1])
    y_current_grade_readiness = np.array([averages['FIRST_CURRENT_GRADE_READINESS'], averages['CURRENT_GRADE_READINESS']])
    y_max_grade_readiness = np.array([averages['FIRST_MAX_GRADE_READINESS'], averages['MAX_GRADE_READINESS']])
    
    # Plot lines and add data labels formatted as percentages
    ax.plot(x_coords, y_current_grade_readiness, label='Actual Growth', marker='o', color='green')
    ax.plot(x_coords, y_max_grade_readiness, label='Expected Growth', marker='o', color='red')
    for x, y in zip(x_coords, y_current_grade_readiness):
        ax.text(x, y, f"{y*100:.1f}%", ha='center', va='bottom', fontsize=slider_fontsize*1.25)
    for x, y in zip(x_coords, y_max_grade_readiness):
        ax.text(x, y, f"{y*100:.1f}%", ha='center', va='bottom', fontsize=slider_fontsize*1.25)

    # Count the number of unique STUDENT_IDs for this month and annotate graph
    num_students = df_filtered['STUDENT_ID'].nunique()
    ax.annotate(f'n = {num_students}', xy=(0, 1), xycoords='axes fraction', 
                xytext=(5, -5), textcoords='offset points', 
                ha='left', va='top', fontsize=slider_fontsize*1.25)

    
    ax.set_xticks(x_coords)
    ax.set_xticklabels(['Placement', 'Current'], fontsize=slider_fontsize*1.25)
    ax.set_title(f'First Month: {month.strftime("%b %Y")}', fontsize=slider_fontsize*1.25)
    ax.set_ylabel('', fontsize=slider_fontsize*1.25)
    ax.set_yticks([])  # Remove y-axis ticks

# Hide unused axes
for j in range(i+1, nrows*ncols):
    axs.flatten()[j].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=slider_fontsize*1.25)
fig.suptitle('', fontsize=slider_fontsize*1.25)

st.pyplot(fig)














































# # Write directly to the app
# st.title("Example Streamlit App :balloon:")
# st.write(
#     """Replace this example with your own code!
#     **And if you're new to Streamlit,** check
#     out our easy-to-follow guides at
#     [docs.streamlit.io](https://docs.streamlit.io).
#     """
# )

# # Get the current credentials
# session = get_active_session()

# # Use an interactive slider to get user input
# hifives_val = st.slider(
#     "Number of high-fives in Q3",
#     min_value=0,
#     max_value=90,
#     value=60,
#     help="Use this to enter the number of high-fives you gave in Q3",
# )

# #  Create an example dataframe
# #  Note: this is just some dummy data, but you can easily connect to your Snowflake data
# #  It is also possible to query data using raw SQL using session.sql() e.g. session.sql("select * from table")
# created_dataframe = session.create_dataframe(
#     [[50, 25, "Q1"], [20, 35, "Q2"], [hifives_val, 30, "Q3"]],
#     schema=["HIGH_FIVES", "FIST_BUMPS", "QUARTER"],
# )

# # Execute the query and convert it into a Pandas dataframe
# queried_data = created_dataframe.to_pandas()

# # Create a simple bar chart
# # See docs.streamlit.io for more types of charts
# st.subheader("Number of high-fives")
# st.bar_chart(data=queried_data, x="QUARTER", y="HIGH_FIVES")

# st.subheader("Underlying data")
# st.dataframe(queried_data, use_container_width=True)




