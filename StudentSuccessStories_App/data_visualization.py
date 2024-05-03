import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patches as patches
import textwrap
import matplotlib.table as mtab # Don't forget this import!
from matplotlib.ticker import FixedLocator




def create_funnel_viz(df,product = '',print_times = False):
    start_time = time.perf_counter()
    # Step 1: All students in the dataset
    all_students = df['STUDENT_ID'].nunique()
    df['ngr_mastered_pct'] = (df['NGR_MASTERED_SKILL_CNT'] + df['NGR_PRIOR_MASTERED_SKILL_CNT']) / df['CURRENT_GRADE_AVAILABLE_CNT']

    # Step 2: Students with TOTAL_PROGRAM_USAGE >= 5 * 3600
    usage_threshold_students_1 = df[df['TOTAL_PROGRAM_USAGE'] >= 1 * 3600]['STUDENT_ID'].nunique()

    # Step 2: Students with TOTAL_PROGRAM_USAGE >= 5 * 3600
    usage_threshold_students = df[df['TOTAL_PROGRAM_USAGE'] >= 5 * 3600]['STUDENT_ID'].nunique()

    # Step 3: Students from step 2 with WEEKLY_USAGE_MIN between 20 and 60 minutes
    weekly_usage_students = df[(df['TOTAL_PROGRAM_USAGE'] >= 5 * 3600) & 
                               (df['WEEKLY_USAGE_MIN'] >= 20) & 
                               (df['WEEKLY_USAGE_MIN'] <= 60)]['STUDENT_ID'].nunique()

    # Step 4: Students from step 3 with CURRENT_GRADE_READINESS >= MAX_GRADE_READINESS - 0.2
    readiness_students = df[(df['TOTAL_PROGRAM_USAGE'] >= 5 * 3600) &
                            (df['WEEKLY_USAGE_MIN'] >= 20) &
                            (df['WEEKLY_USAGE_MIN'] <= 60) &
                            (df['CURRENT_GRADE_READINESS'] >= df['MAX_GRADE_READINESS'] - 0.2)]['STUDENT_ID'].nunique()

    growth_01_students = df[(df['ngr_mastered_pct'] >= 0.1)]['STUDENT_ID'].nunique()

    growth_02_students = df[(df['ngr_mastered_pct'] >= 0.2)]['STUDENT_ID'].nunique()

    growth_05_students = df[(df['ngr_mastered_pct'] >= 0.3)]['STUDENT_ID'].nunique()

    growth_more_05_students = df[(df['ngr_mastered_pct'] >= 0.5)]['STUDENT_ID'].nunique()

    # Preparing data for visualization
    funnel_stages = ['All Students', 'Total Usage >= 1h', 'Total Usage >= 5h', 'Weekly 20-60min', 'NGR Growth >= 10%', 'NGR Growth >= 20%', 'NGR Growth >= 30%', 'NGR Growth >= 50%']
    student_counts = [all_students, usage_threshold_students_1, usage_threshold_students, weekly_usage_students, growth_01_students, growth_02_students, growth_05_students, growth_more_05_students]

    # Plotting the funnel chart
    fig, ax = plt.subplots(figsize=(12, 12))
    bars = ax.barh(funnel_stages, student_counts, color=['blue', 'green', 'orange', 'red'])
    ax.invert_yaxis()  # To display the funnel in the correct order

    # Adding data labels for raw count and percentage
    for bar, count in zip(bars, student_counts):
        percentage = (count / all_students) * 100  # Calculate percentage of all_students
        label = f"{count} ({percentage:.1f}%)"
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2, label, 
                va='center', fontweight='bold')

    plt.title(f"Student Progression Funnel - {product}")
    plt.xlabel('Number of Students')
    plt.tight_layout()
    st.pyplot(fig)
    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The create_funnel_viz took {duration} seconds to run")

def plot_kids_on_track(df, split_by=None, size_font=16, product = 'my_reading_academy',type = 'score',print_times = False):
    """
    Plots the percentage of kids on track across time, with an option to split the graph by a specified column.

    Parameters:
    - df: DataFrame containing the required columns for plotting.
    - split_by: Optional; column name by which to split the graph. If provided, creates separate graphs for each unique value in this column.
    - size_font: Font size for the labels and legend text. Default is 10.
    """
    start_time = time.perf_counter()
    def _plot_status_graph(df, title_suffix=''):
        """
        Helper function modified to use status_labels and status_colors, and to include student count annotations in a dedicated space.
        """

        status_labels = {
            'Ahead': 'Next Grade Ready',
            'ClosedGap': 'Caught Up',
            'On Track': 'On Track',
            'CatchingUp': 'Narrowed the Gap',
            'Behind': 'Below Target'
        }
        status_colors = {
            'Ahead': '#69D112', # Completed this month light green
            'On Track': '#167CF1', # Educator Center Blue
            'ClosedGap': '#0342EC', # AofL Blue
            'CatchingUp': '#FCB800', # Schools Yellow
            'Behind': '#F48A3E' # Support Orange
        }

        # Status is defined in process_monthly_growth_data()

        categories = ['At Placement', 'Current Status']  # Removed the placeholder for "Total Students"
        status_counts = df.groupby('Status')['STUDENT_ID'].nunique().reindex(status_labels.keys(), fill_value=0)
        first_status_counts = df.groupby('First_Status')['STUDENT_ID'].nunique().reindex(status_labels.keys(), fill_value=0)

        data = np.array([first_status_counts, status_counts])
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjusted size for readability and extra annotation space
        bottom = np.zeros(len(categories))
        bar_width = 0.8  # Adjust this value as needed to make the bars thinner or thicker

        for status, color in reversed(list(status_colors.items())):
            i = list(status_labels.keys()).index(status)
            bars = ax.bar(categories, data[:, i], bottom=bottom, label=status_labels[status], color=color, width=bar_width)  # Specify the width here
            bottom += data[:, i]

            for bar, count in zip(bars, data[:, i]):
                percentage = count / data.sum(axis=1)[i % len(categories)] * 100
                if percentage >= 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f'{percentage:.0f}%', 
                            ha='center', va='center', color='white', fontsize=size_font*2)


        # # Annotations calculations
        # total_students = df['STUDENT_ID'].nunique()
        # avg_weekly_minutes = df['WEEKLY_USAGE_MIN'].median()
        # avg_active_weeks = df['ACTIVE_WEEKS'].median()
        # avg_grade_readiness_diff = (df['CURRENT_GRADE_READINESS'] - df['FIRST_CURRENT_GRADE_READINESS']).median() * 100  # Convert to percentage

        # if True: # title_suffix == '':
        #     # Positioning annotations
        #     annotations = [
        #         f'Total Students: {total_students}',
        #         f'Avg. Minutes\nPer Week: {avg_weekly_minutes:.0f}',
        #         f'Avg. Active\nWeeks: {avg_active_weeks:.0f}',
        #         f'Avg. Progress\nToward Readiness : {avg_grade_readiness_diff:.0f}%'
        #     ]

        #     # Adjusting the plot to accommodate annotations
        #     for i, annotation in enumerate(annotations):
        #         ax.text(len(categories) -1.5, ax.get_ylim()[1] * (0.90 - i*0.25), annotation,
        #                 ha='center', va='center', color='black', fontsize=size_font*1.25,
        #                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#D8D8D8', edgecolor='none', alpha=1))


        ax.set_ylim(0, ax.get_ylim()[1]*1.15)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=size_font)  # Change '14' to your desired font size


        legend_elements = [Patch(facecolor=status_colors[status], label=status_labels[status]) for status in reversed(status_labels.keys())]
        ax.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=size_font*0.9)

        ax.set_title(f'Progress Toward Next Grade Readiness{title_suffix}',fontsize = size_font*1.25)
        st.pyplot(fig)

    # Check if splitting the graph is required
    if split_by and split_by in df.columns:
        unique_values = df[split_by].dropna().unique()
        grade_labels = {
            'Pre-K' : 'Pre-K',
            'Kindergarten' : 'Kindergarten',
            '1' : '1st Grade',
            '2' : '2nd Grade',
            '3' : '3rd Grade',
            'Other' : 'Other'
        }
        unique_values_sorted = sorted(unique_values, key=lambda x: list(grade_labels).index(x))
        # Create a separate graph for each unique value in the split_by column
        for value in unique_values_sorted:
            df_filtered = df[df[split_by] == value]
            try:
                _plot_status_graph(df_filtered, title_suffix=f" - {grade_labels.get(value, 'Unknown Grade')}\n{product}")
            except Exception as e:
                # If an error occurs, print the error and continue
                print(f"An error occurred while plotting {grade_labels.get(value, 'Unknown Grade')}: {e}")
    else:
        # Plot a single graph for the entire dataset
        try:
            _plot_status_graph(df, title_suffix=f"\n{product}")
        except Exception as e:
            print(f"An error occurred while plotting: {e}")
    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The plot_kids_on_track took {duration} seconds to run")
def plot_growth_status(df, slider_fontsize=16,product = 'My Math Academy'):
    st.write("")  # For spacing

    # Calculate the total number of unique students in the original dataframe
    total_students = df['STUDENT_ID'].nunique()
    if total_students == 0:
        return None

    avg_weekly_minutes = df['WEEKLY_USAGE_MIN'].median()
    avg_active_weeks = df['ACTIVE_WEEKS'].median()
    avg_grade_readiness_diff = (df['CURRENT_GRADE_READINESS'] - df['FIRST_CURRENT_GRADE_READINESS']).median() * 100  # Convert to percentage
    progress_sentence = f"By using {product} for {avg_weekly_minutes:.0f} minutes per week, for {avg_active_weeks:.0f} weeks, the average student is expected to achieve {avg_grade_readiness_diff:.0f}% progress toward readiness."

    # Define the dataframes
    dataframes = [
        df,  # The entire dataset for the total overview
        df[df['Status'] == 'Ahead'],
        df[df['Status'] == 'ClosedGap'],
        df[df['Status'] == 'CatchingUp']
    ]

    # Labels for each condition
    labels = [
        progress_sentence,  # The label for the new box
        ">90% mastery of current and prior grade level skills",
        "Caught up to grade level after starting behind",
        "Narrowed the gap to grade level after starting behind"
    ]

    # Setup figure and axes for the grid
    fig, axs = plt.subplots(4, 2, figsize=(12, 12), dpi=100)  # Adjusted for 4 rows

    for i, (data_frame, label) in enumerate(zip(dataframes, labels)):
        unique_students = data_frame['STUDENT_ID'].nunique()
        percentage = 100 if i == 0 else (unique_students / total_students) * 100  # Full percentage for the total overview

        # Create a rounded rectangle (blue box) in the first column
        box = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.1,rounding_size=0.1", 
                                     color='#0342EC', transform=axs[i, 0].transAxes)
        axs[i, 0].add_patch(box)
        if i == 0:
            # Special formatting for the first box
            axs[i, 0].text(0.5, 0.6, f'{total_students}', ha='center', va='center', fontsize=slider_fontsize*5, color='white', weight='bold', transform=axs[i, 0].transAxes)
            axs[i, 0].text(0.5, 0.3, 'students', ha='center', va='center', fontsize=slider_fontsize*4, color='white', transform=axs[i, 0].transAxes)
        else:
            axs[i, 0].text(0.5, 0.6, f'{percentage:.0f}%', ha='center', va='center', fontsize=slider_fontsize*5, color='white', weight='bold', transform=axs[i, 0].transAxes)
            axs[i, 0].text(0.5, 0.3, f'{unique_students} students', ha='center', va='center', fontsize=slider_fontsize*4, color='white', transform=axs[i, 0].transAxes)

        axs[i, 0].set_xlim(0, 1)
        axs[i, 0].set_ylim(0, 1)
        axs[i, 0].axis('off')

        # Second column for the labels
        if i == 0:
            axs[i, 1].text(0.5, 0.5, label, ha='center', va='center', fontsize=slider_fontsize*2.4, wrap=True)
            axs[i, 1].axis('off')
        else:
            axs[i, 1].text(0.5, 0.5, label, ha='center', va='center', fontsize=slider_fontsize*3, wrap=True)
            axs[i, 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
def plot_skill_domain_distribution(df, split_by=None, AOFL_PRODUCT = '',num_domains = 5,title_suffix_top = '',sizefont = 16,print_times = False):
    start_time = time.perf_counter()
    def plot_graph(data, title_suffix=''):
        processed_data = data.copy()
        # Step 1: Calculate the required sums and total
        # Adding new columns to the DataFrame for completed skills and total skills
        processed_data['NGR_TOTAL_COMPLETED'] = processed_data['NGR_COMPLETED_SKILLS_BY_DOMAIN'] + processed_data['NGR_PRIOR_COMPLETED_SKILLS_BY_DOMAIN']
        processed_data['Total_Skills'] = processed_data['NGR_PRIOR_KNOWLEDGES_BY_DOMAIN'] + processed_data['NGR_TOTAL_COMPLETED'] + processed_data['NGR_REMAIN_SKILLS_BY_DOMAIN']


        # Calculate the average values for each component as percentages of the total
        processed_data['Avg_Prior_Knowledge_%'] = (processed_data['NGR_PRIOR_KNOWLEDGES_BY_DOMAIN'] / processed_data['Total_Skills']) * 100
        processed_data['Avg_Completed_%'] = (processed_data['NGR_TOTAL_COMPLETED'] / processed_data['Total_Skills']) * 100
        processed_data['Avg_Remain_%'] = (processed_data['NGR_REMAIN_SKILLS_BY_DOMAIN'] / processed_data['Total_Skills']) * 100


        # Step 2: Group by SKILL_DOMAIN and calculate the mean percentage for each component
        domain_percentages = processed_data.groupby('SKILL_DOMAIN').agg({
            'Avg_Prior_Knowledge_%': 'mean',
            'Avg_Completed_%': 'mean',
            'Avg_Remain_%': 'mean'
        }).reset_index()

        # First, separate the 'Grand Total' row
        grand_total_row = domain_percentages[domain_percentages['SKILL_DOMAIN'] == 'Grand Total']

        # Then, get the rest of the dataframe without 'Grand Total'
        rest_of_df = domain_percentages[domain_percentages['SKILL_DOMAIN'] != 'Grand Total']

        # Sort the rest of the dataframe by 'Avg_Completed_%' in descending order
        rest_of_df_sorted = rest_of_df.sort_values(by='Avg_Completed_%', ascending=False)

        # Concatenate the 'Grand Total' row at the top and the sorted dataframe
        # sorted_domain_percentages = pd.concat([grand_total_row, rest_of_df_sorted]).head(6) # Remove Grand Total Row
        sorted_domain_percentages = rest_of_df_sorted.head(num_domains)

        # filename = f"{AOFL_PRODUCT}_{title_suffix.replace(' ', '_')}.csv".strip("_")
        # sorted_domain_percentages.to_csv(filename)

        # Prepare data for stacked bar chart using the adjusted dataframe
        categories = sorted_domain_percentages['SKILL_DOMAIN'].tolist()
        prior_knowledge_vals = sorted_domain_percentages['Avg_Prior_Knowledge_%'].values
        completed_vals = sorted_domain_percentages['Avg_Completed_%'].values
        remain_vals = sorted_domain_percentages['Avg_Remain_%'].values

        color_scheme = {
            'Prior Knowledge': '#EDF887',
            'Completed': '#69D112',
            'Remaining': '#D8D8D8'
        }

        # Plotting adjustments
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.barh(categories, prior_knowledge_vals, color=color_scheme['Prior Knowledge'], edgecolor='white', label='Prior Knowledge')
        ax.barh(categories, completed_vals, left=prior_knowledge_vals, color=color_scheme['Completed'], edgecolor='white', label='Completed')
        bottom_second = prior_knowledge_vals + completed_vals
        ax.barh(categories, remain_vals, left=bottom_second, color=color_scheme['Remaining'], edgecolor='white', label='Remaining')


        # Adding data labels with condition for values greater than 5%
        for i, (category, pk_val, comp_val, remain_val) in enumerate(zip(categories, prior_knowledge_vals, completed_vals, remain_vals)):
            if pk_val > 5:
                ax.text(pk_val/2, i, f"{pk_val:.0f}%", va='center', ha='center', color='black', fontsize=sizefont*1.5)
            if comp_val > 5:
                ax.text(pk_val + comp_val/2, i, f"{comp_val:.0f}%", va='center', ha='center', color='black', fontsize=sizefont*1.5)
            if remain_val > 5:
                ax.text(pk_val + comp_val + remain_val/2, i, f"{remain_val:.0f}%", va='center', ha='center', color='black', fontsize=sizefont*1.5)

        # Customize the plot
        ax.invert_yaxis()
        ax.set_ylabel('')  # Label y-axis
        ax.set_title(f'Skill Domain Progress {title_suffix}',fontsize = sizefont*1.5)
        ax.legend(loc='upper center', ncol=3,fontsize = sizefont*1.25)
        plt.subplots_adjust(top=1.375)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xlabel('')  # Remove x-axis label

        def wrap_text(text, width):
            """A simple function to wrap text at a given width."""
            return '\n'.join(textwrap.wrap(text, width))
        # Customize SKILL_DOMAIN y-axis labels with increased font size and wrapped text
        wrapped_labels = [wrap_text(label, 30) for label in categories]  # Adjust 20 based on your needs
        locations = ax.get_yticks()

        # Set the tick locations with FixedLocator
        ax.yaxis.set_major_locator(FixedLocator(locations))

        # Set the custom labels with set_yticklabels at those locations
        ax.set_yticklabels(wrapped_labels, fontsize=sizefont*1.5)  # Adjust fontsize as needed

        st.pyplot(fig)
    
    if split_by and split_by in df.columns:
        unique_values = df[split_by].dropna().unique()
        grade_labels = {
            'Pre-K' : 'Pre-K',
            'Kindergarten' : 'Kindergarten',
            '1' : '1st Grade',
            '2' : '2nd Grade',
            '3' : '3rd Grade',
            'Other' : 'Other'
        }
        # Sort unique_values based on the order in grade_labels
        unique_values_sorted = sorted(unique_values, key=lambda x: list(grade_labels).index(x))
        for value in unique_values_sorted:
            df_filtered = df[df[split_by] == value]
            try:
                # Try to plot the graph
                plot_graph(df_filtered, title_suffix=f" - {grade_labels.get(value, 'Unknown Grade')}\n{AOFL_PRODUCT}")
            except Exception as e:
                # If an error occurs, print the error and continue
                print(f"An error occurred while plotting {grade_labels.get(value, 'Unknown Grade')}: {e}")
    else:
        if title_suffix_top == '':
            title_suffix = f"\n{AOFL_PRODUCT}"
        else:
            title_suffix = title_suffix_top
        try:
            plot_graph(df, title_suffix=title_suffix)
            
        except Exception as e:
            print(f"An error occurred while plotting: {e}")
    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The plot_skill_domain_distribution took {duration} seconds to run")




# Function to render table
def render_mpl_table(df, col_width=2.0, row_height=0.625, font_size=14,
                     header_color='#0342EC', row_colors=['#f1f1f2', '#ffffff'], edge_color='w',
                     bbox=[0, 0, 1, 1], axs=None, **kwargs):
    def aggregate_data(df, groupby_columns):
            """
            Aggregates the DataFrame by the given list of groupby_columns.
            Calculates the count of unique STUDENT_IDs, and the averages of WEEKLY_USAGE_MIN,
            TOTAL_PROGRAM_USAGE, FIRST_NGR_BYPASSED_PCT, and CURRENT_GRADE_READINESS.
            """
            aggregation = {
                'STUDENT_ID': pd.Series.nunique,
                'WEEKLY_USAGE_MIN': 'mean',
                'TOTAL_PROGRAM_USAGE': 'mean',
                'FIRST_NGR_BYPASSED_PCT': 'mean',
                'NGR_TOTAL_MASTERED_SKILL_CNT' : 'mean',
                'CURRENT_GRADE_READINESS': 'mean'
            }
        
            # Group by the provided list of columns and apply the aggregation
            aggregated_df = df.groupby(groupby_columns).agg(aggregation)
            
            # Resetting index if you want to turn the grouping columns into regular columns
            aggregated_df = aggregated_df.reset_index()
        
            # Rounding and converting columns
            aggregated_df['WEEKLY_USAGE_MIN'] = round(aggregated_df['WEEKLY_USAGE_MIN'],0)
            aggregated_df['TOTAL_PROGRAM_USAGE'] = aggregated_df['TOTAL_PROGRAM_USAGE'] / 3600
            aggregated_df['FIRST_NGR_BYPASSED_PCT'] = aggregated_df['FIRST_NGR_BYPASSED_PCT']
            aggregated_df['CURRENT_GRADE_READINESS'] = aggregated_df['CURRENT_GRADE_READINESS']
        
            return aggregated_df[aggregated_df['STUDENT_ID'] > 0]
    # Function to draw mini-bar
    def draw_mini_bar(ax, x, y, width, height, percentage):
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor='white', edgecolor='white'))
        ax.add_patch(plt.Rectangle((x, y), width * percentage, height, facecolor='#2CA22C', edgecolor='white'))
        # this labels the red bar on each barchart
        ax.text(x + 0.005, y + (height / 2), f"{percentage:.0%}", color='w', ha='left', va='center',fontsize = 12)
    
    data = aggregate_data(df, groupby_columns=['AOFL_PRODUCT', 'ORGANIZATION_NAME', 'STUDENT_GRADE', 'USAGE_CATEGORY'])

    
    # Number of rows and columns
    num_rows, num_cols = data.shape
    
    if axs is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, axs = plt.subplots(figsize=size)
        axs.axis('off')
    
    mpl_table = mtab.table(ax=axs, cellText=list([[str(j) for j in l] for l in data.values.tolist()]), 
                           bbox=bbox, colLabels=data.columns, **kwargs)
    
    # This creates a dictionary of all cells in the table. We can use it to access and manipulate 
    # individual cells. The keys are the row/column indices of each cell
    dictOfCells = mpl_table.get_celld()
    
    # looping through the keys from the dictionary of cells
    for i in dictOfCells.keys():
        cell = dictOfCells[i]
        cell.set_edgecolor(edge_color)
        cell.set_fontsize(font_size)  # Set font size for each cell's text

        
        if i[0] == 0: # Sets the background color for the header bar
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else: # Sets the background color for the table, making it switch colors every row
            cell.set_facecolor(row_colors[(i[0]) % len(row_colors)]) # Set the color
            cell.set(xy=((1 - (1/num_cols)), (i[0]-1)/(num_rows + 1))) # Dynamic to change based on table size
            
        cell.set(height=(1/(num_rows + 1))) # Changes based on table size
        # st.wrte(f'Height:{(1/(num_rows + 1))}')
        # DRAW BARS IN LAST COLUMN
        if i[0]>0 and i[1] == len(data.columns)-1:
            # x and y positions that were set above
            xPos = cell.get_x()
            yPos = (num_rows - i[0]) / (num_rows + 1)
            percentage = float(cell.get_text().get_text())
            # notice that the percentage parameter is set to the value in each cell that the barchart is in
            draw_mini_bar(axs, xPos, yPos, cell.get_width(), cell.get_height(), percentage)
    st.pyplot(fig)

def aggregated_table(df, groupby_columns, sort_by = None, format = True):
            """
            Aggregates the DataFrame by the given list of groupby_columns.
            Calculates the count of unique STUDENT_IDs, and the averages of WEEKLY_USAGE_MIN,
            TOTAL_PROGRAM_USAGE, ngr_bypassed_pct, and CURRENT_GRADE_READINESS.
            """
            aggregation = {
                'STUDENT_ID': pd.Series.nunique,
                'WEEKLY_USAGE_MIN': 'mean',
                'TOTAL_PROGRAM_USAGE': 'mean',
                'ngr_bypassed_pct': 'mean',
                'CURRENT_GRADE_READINESS': 'mean',
                'NGR_TOTAL_MASTERED_SKILL_CNT' : 'mean'
                
            }
        
            # Group by the provided list of columns and apply the aggregation
            aggregated_df = df.groupby(groupby_columns).agg(aggregation)
            
            # Resetting index if you want to turn the grouping columns into regular columns
            aggregated_df = aggregated_df.reset_index()
        
            if format:
                def format_percentage(df, column_name):
                    return df[column_name].apply(lambda x: f"{x}%")
                # Rounding and converting columns
                aggregated_df['WEEKLY_USAGE_MIN'] = round(aggregated_df['WEEKLY_USAGE_MIN'], 0)
                aggregated_df['TOTAL_PROGRAM_USAGE'] = round(aggregated_df['TOTAL_PROGRAM_USAGE'] / 3600,0)  # Convert seconds to hours
                aggregated_df['ngr_bypassed_pct'] = round(aggregated_df['ngr_bypassed_pct'] * 100,0)
                aggregated_df['CURRENT_GRADE_READINESS'] = round(aggregated_df['CURRENT_GRADE_READINESS'] * 100,0)
                aggregated_df['NGR_TOTAL_MASTERED_SKILL_CNT'] = round(aggregated_df['NGR_TOTAL_MASTERED_SKILL_CNT'], 0)

                # aggregated_df['ngr_bypassed_pct'] = format_percentage(aggregated_df,'ngr_bypassed_pct')

            # Dictionary for renaming columns to more readable names
            column_renames = {
                'STUDENT_ID': 'Number of Students',
                'WEEKLY_USAGE_MIN': 'Average Weekly Usage (min)',
                'TOTAL_PROGRAM_USAGE': 'Total Program Usage (hours)',
                'ngr_bypassed_pct': 'Placement Grade Readiness (%)',
                'CURRENT_GRADE_READINESS': 'Current Grade Readiness (%)',
                'NGR_TOTAL_MASTERED_SKILL_CNT': 'Average Skills Mastered per Student',
                'SCHOOL_NAME' : 'School',
                'STUDENT_GRADE' : 'Grade',
                'CLASSROOM_NAME' : 'Classroom'
            }
            
            # Remove rows with missing values in groupby columns
            # aggregated_df.dropna(subset=groupby_columns, inplace=True)

            # Rename columns according to the dictionary
            aggregated_df.rename(columns=column_renames, inplace=True)

            filtered_sorted = aggregated_df[aggregated_df['Number of Students'] > 0].sort_values(by = 'Number of Students', ascending = [False])

            # filtered_sorted = st.dataframe(filtered_sorted.set_index(filtered_sorted.columns[0]))
            
            # Filter out any groups where no students were counted
            return filtered_sorted
def create_dumbbell_plot(df, groupby='SCHOOL_NAME'):
    # Ensure the columns exist in the DataFrame
    required_columns = ['ngr_bypassed_pct', 'CURRENT_GRADE_READINESS']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must include the columns: {required_columns}")

    # Grouping the DataFrame by the specified groupby parameter and calculating the mean
    group_data = df.groupby(groupby)[required_columns].mean().reset_index()

    # Plotting
    fig, ax = plt.subplots()
    for index, row in group_data.iterrows():
        start_point = row['ngr_bypassed_pct']
        end_point = row['CURRENT_GRADE_READINESS']
        
        # Draw the line
        ax.plot([start_point, end_point], [index, index], color='#69D112', zorder=1)
        
        # Draw the start and end points
        ax.scatter([start_point], [index], color='#EDF887', edgecolor='black', s=300, zorder=2, label='Prior Knowledge' if index == 0 else "")
        ax.scatter([end_point], [index], color='#69D112', edgecolor='black', s=300, zorder=2, label='Completed' if index == 0 else "")
        
        # Add data labels
        ax.text(start_point, index + 0.2, f'{start_point:.0%}', fontsize=12, ha='left', va='bottom')
        ax.text(end_point, index + 0.2, f'{end_point:.0%}', fontsize=12, ha='left', va='bottom')
    
    # Adding legend at the top
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # Removing x-axis ticks
    ax.xaxis.set_ticks([])

    # Adding labels and title
    ax.set_yticks(range(len(group_data)))
    ax.set_yticklabels(group_data[groupby])
    ax.set_title('Dumbbell Plot of Prior Knowledge and Completion Readiness by ' + groupby)

    st.pyplot(fig)