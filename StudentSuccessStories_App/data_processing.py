import streamlit as st
import time
import numpy as np
import pandas as pd

def combine_multiple_rosterings(df, grouping_cols=['AOFL_PRODUCT', 'STUDENT_ID', 'MONTH'],print_times = False):
    """
    Aggregates the DataFrame based on the provided grouping columns.
    Performs a distinct listagg for string columns, takes the maximum for numeric and datetime columns.
    
    Why? Because a student could be in multiple classrooms or schools, so this limits the 
    data duplication.
    """
    start_time = time.perf_counter()
    # Identify string, numeric, and datetime columns
    min_cols = ['FIRST_MONTH','FIRST_CURRENT_GRADE_READINESS','FIRST_MAX_GRADE_READINESS','FIRST_NGR_BYPASSED_PCT']
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime']).columns.tolist()
    
    # Define aggregation functions
    agg_funcs = {}
    for col in string_cols:
         if col not in grouping_cols:
            agg_funcs[col] = lambda x: '; '.join(np.sort(x.dropna().unique().astype(str)))
    for col in numeric_cols + datetime_cols:
        if col not in grouping_cols:
            if col in min_cols:
                agg_funcs[col] = 'min'
            else:
                agg_funcs[col] = 'max'
    # Perform aggregation
    processed_df = df.groupby(grouping_cols, as_index=False).agg(agg_funcs)
    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The combine_multiple_rosterings took {duration} seconds to run")
    return processed_df
def process_monthly_growth_data(data_raw,data_first_raw, product,slider_values,print_times = False):
    """
    Merges and processes monthly growth data from raw data frames, incorporating initial assessment data
    to compute growth metrics and filter based on specific criteria. It calculates total bypassed skill count,
    categorizes student grades, computes active weeks and growth metrics, then filters the merged dataset based on
    pre-defined criteria to yield a subset of the data for further analysis or visualization.

    Parameters:
    - data_raw: pandas DataFrame containing the raw monthly growth data.
    - data_first_raw: pandas DataFrame containing the initial assessment data for each student.
    - product: String specifying the name of the product for which the data is being processed.

    Processes:
    1. Deduplicates both `data_raw` and `data_first_raw` using a custom rostering method.
    2. Merges the deduplicated data frames on 'STUDENT_ID', keeping specific initial assessment variables.
    3. Calculates the total bypassed skill count by summing current and prior bypassed skills.
    4. Categorizes 'STUDENT_GRADE' within a predefined order for analysis.
    5. Computes 'ACTIVE_WEEKS' as the ratio of total program usage to weekly usage minutes.
    6. Calculates growth metrics ('NGR_GROWTH', 'NGR_TARGET_GROWTH') based on grade readiness from initial assessment to current assessment.
    7. Applies a set of filters to the merged data to obtain a subset based on criteria such as grade readiness, weekly usage minutes, and total program usage.

    Returns:
    - filtered_data: The filtered DataFrame after applying adjustments, categorizations, calculations, and filters, ready for analysis or reporting.

    Outputs:
    - Prints a message with the product name, initial, and filtered unique student counts.
    """
    start_time = time.perf_counter()
    start_time_2 = time.perf_counter()
    
    data = combine_multiple_rosterings(data_raw) # Perform deduplication
    data_first = combine_multiple_rosterings(data_first_raw,['STUDENT_ID'])
    end_time_2 = time.perf_counter()
    duration_2 = round(end_time_2 - start_time_2,3)
    if print_times:
        st.write(f"The filter took {duration_2} seconds to run")
    
    filter_premerge = (
        (data['WEEKLY_USAGE_MIN'] >= slider_values['slider_mins_week'])
        & (data['WEEKLY_USAGE_MIN'] <= slider_values['slider_mins_week_max'])
        & (data['TOTAL_PROGRAM_USAGE'] >= slider_values['slider_total_hours'] * 3600)
        # & ((data['CURRENT_GRADE_READINESS'] >= data['MAX_GRADE_READINESS'] - 0.2))
    )

    filtered_data_1 = data[filter_premerge].copy()

    filtered_out_premerge = data[~filter_premerge].copy()
    if print_times:
        st.write("Rows filtered out pre-merge:")
        st.write(filtered_out_premerge)

    vars_to_keep = ['STUDENT_ID','FIRST_MONTH','FIRST_CURRENT_GRADE_READINESS','FIRST_MAX_GRADE_READINESS','FIRST_NGR_BYPASSED_PCT']
    merged = pd.merge(left=filtered_data_1, 
                         right=data_first[vars_to_keep], 
                         how='left', 
                         left_on=['STUDENT_ID'],
                         right_on=['STUDENT_ID'])
    
    # Create some new fields that are used later in the visualizations
    merged['NGR_TOTAL_BYPASSED_SKILL_CNT'] = merged['NGR_BYPASSED_SKILL_CNT'] + merged['NGR_PRIOR_BYPASSED_SKILL_CNT']
    merged['NGR_TOTAL_MASTERED_SKILL_CNT'] = merged['NGR_MASTERED_SKILL_CNT'] + merged['NGR_PRIOR_MASTERED_SKILL_CNT']
    merged['ngr_mastered_pct'] = merged['NGR_TOTAL_MASTERED_SKILL_CNT'] / merged['CURRENT_GRADE_AVAILABLE_CNT']
    merged['ngr_bypassed_pct'] = merged['NGR_TOTAL_BYPASSED_SKILL_CNT'] / merged['CURRENT_GRADE_AVAILABLE_CNT']
    grades_order = ['Under Pre-K','Pre-K','Kindergarten', '1', '2','3','Other']
    merged['STUDENT_GRADE'] = pd.Categorical(merged['STUDENT_GRADE'], categories=grades_order, ordered=True)
    merged['ACTIVE_WEEKS'] = (merged['TOTAL_PROGRAM_USAGE'] / merged['WEEKLY_USAGE_MIN']) / 60
    merged['NGR_GROWTH'] = merged['CURRENT_GRADE_READINESS'] - merged['ngr_bypassed_pct']
    merged['NGR_TARGET_GROWTH'] = merged['MAX_GRADE_READINESS'] - merged['FIRST_MAX_GRADE_READINESS']
    merged['CURRENT_GAP'] = merged['MAX_GRADE_READINESS'] - merged['CURRENT_GRADE_READINESS'] 
    merged['FIRST_GAP'] = merged['FIRST_MAX_GRADE_READINESS'] - merged['ngr_bypassed_pct'] # Classify based on current bypassed skills
    merged['USAGE_CATEGORY'] = pd.cut(merged['WEEKLY_USAGE_MIN'], 
                                      bins=[0, 30, 45, 60, float('inf')], 
                                      labels=['>0-30', '>30-45', '>45-60', '>60'], 
                                      right=False)
    
    # Create Growth Status
    def calculate_status(current, target,first_gap = 0,current_gap = 0):
            if current >= 0.90:         # Finished 90% of content
                return 'Ahead'
            elif current >= target:     # On Track has two sub-status
                if first_gap > 0:
                    return 'ClosedGap'  # Was behind and caught up
                else:
                    return 'On Track'   # Was on Track both times. 
            elif current_gap < first_gap:
                return 'CatchingUp'     # Still behind, but less behind
            else:
                return 'Behind'         # Everyone one
    
    # Setting the calculated statuses, as before
    merged.loc[:, 'Status'] = merged.apply(lambda row: calculate_status(row['CURRENT_GRADE_READINESS'], row['MAX_GRADE_READINESS'],row['FIRST_GAP'],row['CURRENT_GAP']), axis=1)
    merged.loc[:, 'First_Status'] = merged.apply(lambda row: calculate_status(row['FIRST_NGR_BYPASSED_PCT'], row['FIRST_MAX_GRADE_READINESS']), axis=1)


    # correlation = merged['NGR_GROWTH'].corr(merged['ngr_mastered_pct'])
    # st.write(f"Correlation Coefficient: {correlation}")
    # st.write(merged[merged['NGR_GROWTH'] != merged['ngr_mastered_pct']])
    
    # Apply filters based on the provided parameters
    filter_postmerge = (
        (merged['ACTIVE_WEEKS'] >= slider_values['slider_active_weeks'])
        & ((merged['CURRENT_GRADE_READINESS'] >= 0.9) | (merged['ngr_mastered_pct'] >= slider_values['slider_ngr_growth']))
    )
    filtered_data = merged[filter_postmerge].copy()

    filtered_out_postmerge = merged[~filter_postmerge].copy()
    if print_times:
        st.write("Rows filtered out post-merge:")
        st.write(filtered_out_postmerge)
    
    # Calculate unique student counts
    initial_unique_count = data['STUDENT_ID'].nunique()
    filtered_unique_count = filtered_data['STUDENT_ID'].nunique()
    
    # Print the message with unique counts and the product name
    st.write(f"Product: {product}. Initial unique student count: {initial_unique_count}. Filtered unique student count: {filtered_unique_count}.")

    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The process_monthly_growth_data took {duration} seconds to run")
    return filtered_data
def process_skill_data(original_df,print_times = False):
    """
    Processes skill data by first handling duplicate classes using aggregate_student_data function
    and then summing specific metrics. Adds a grand total row for each STUDENT_ID across all SKILL_DOMAINs.
    """
    start_time = time.perf_counter()

    df = original_df.copy()

    #  # Create a copy of the SKILL_DOMAIN column to preserve the full skill domain names
    df['Skill Domain Full'] = df['SKILL_DOMAIN'].values
    # Define your dictionary for SKILL_DOMAIN aliases
    skill_domain_aliases = {
        'Classify objects and count the number of objects in each category.': 'Classify and Count Objects',
        'Reason with shapes and their attributes.':'Shapes and attributes',
        'Identify and describe shapes (squares, circles, triangles, rectangles, hexagons, cubes, cones, cylinders, and spheres).':'Identify and describe 2D and 3D shapes',
        'Understand and apply properties of operations and the relationship between addition and subtraction.':'Properties of operations',
        'Work with addition and subtraction equations.':'Work with equations',
        'Use place value understanding and properties of operations to add and subtract.':'Apply strategies to add and subtract',
        'Relate addition and subtraction to length.':'Relate operations to length',
        'Use place value understanding and properties of operations to add and subtract.':'Apply strategies to add and subtract',
        'Describe and compare measurable attributes.':'Measurable attributes',
        'Understand addition as putting together and adding to, and understand subtraction as taking apart and taking from.':'Understanding addition and subtraction',
        'Work with numbers 11?19 to gain foundations for place value.':'Work With Numbers 11-19',
        'Reading Strategies':'Metacognitive Skills',
        'Comprehension and Collaboration':'Reading Comprehension Across Genres',
        'Range of Reading and Level of Text Complexity':'Reading Complex Text',
        'Expository Text':'Reading Comprehension (Informational)',
        'Literary Text':'Reading Comprehension (Literary)'
        # Add more aliases as needed
    }
    
    # Replace SKILL_DOMAIN values in place using the aliases dictionary
    df['SKILL_DOMAIN'] = df['SKILL_DOMAIN'].map(skill_domain_aliases).fillna(df['SKILL_DOMAIN']).apply(lambda x: x.title())

    # Initial aggregation to handle duplicate classes
    initial_grouping_cols = ['STUDENT_ID', 'SKILL_DOMAIN', 'SKILL_GRADE']
    df_agg = combine_multiple_rosterings(df, grouping_cols=initial_grouping_cols)

    # Columns for further processing
    sum_cols = ['COMPLETED_SKILLS_BY_DOMAIN', 'PRIOR_KNOWLEDGES_BY_DOMAIN', 
                'PRIOR_COMPLETED_SKILLS_BY_DOMAIN', 'REMAIN_SKILLS_BY_DOMAIN', 
                'NGR_COMPLETED_SKILLS_BY_DOMAIN', 'NGR_PRIOR_KNOWLEDGES_BY_DOMAIN',
                'NGR_PRIOR_COMPLETED_SKILLS_BY_DOMAIN', 'NGR_REMAIN_SKILLS_BY_DOMAIN',
                'IS_STRUGGLING_SKILLS_BY_DOMAIN']
    max_cols = [col for col in df_agg.columns if col not in sum_cols + initial_grouping_cols]
    
    # Second aggregation without SKILL_GRADE and with summing specific metrics
    df_agg_2 = df_agg.groupby(['STUDENT_ID', 'SKILL_DOMAIN'], as_index=False).agg(
        {**{col: 'sum' for col in sum_cols}, **{col: 'max' for col in max_cols}}
    )
    
    # Calculate grand totals for each STUDENT_ID
    grand_totals = df_agg_2.groupby('STUDENT_ID', as_index=False).agg(
        {**{col: 'sum' for col in sum_cols}, **{col: 'max' for col in ['STUDENT_ID'] + max_cols}}
    )
    grand_totals['SKILL_DOMAIN'] = 'Grand Total'  # Indicating these rows are grand totals
    
    # Combine grand totals back with the aggregated data
    df_final_with_totals = pd.concat([df_agg_2, grand_totals], ignore_index=True).sort_values(['STUDENT_ID', 'SKILL_DOMAIN'])

    end_time = time.perf_counter()
    duration = round(end_time - start_time,3)
    if print_times:
        st.write(f"The process_skill_data took {duration} seconds to run")
    return df_final_with_totals