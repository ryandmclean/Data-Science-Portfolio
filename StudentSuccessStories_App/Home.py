import streamlit as st
from _initialize import initialize_app
from snowpark import get_active_session
state, url_params = initialize_app()
session = get_active_session()

# Import python packages
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_extras.app_logo import add_logo
import time
from data_processing import process_monthly_growth_data, process_skill_data
from data_visualization import create_funnel_viz, plot_kids_on_track,plot_growth_status,plot_skill_domain_distribution,render_mpl_table,aggregated_table,create_dumbbell_plot
from data_query import build_slider_menu,query_snowflake,build_snowflake_queries

# Get the current credentials
print_times = False
dev_mode = False

def main():
    col1, col2 = st.columns(2)
    with col1:
        st.title('Mastery Student Exemplars')
    with col2:
        st.image('AofL_LG_white-on-blue-rounded_icon_RGB.png', width = 150)
    

    slider_values = build_slider_menu(session)
    
    sf_queries = build_snowflake_queries(slider_values)

    slider_fontsize = 12

    if st.button('Press to Execute'):
        ###################################
        # Query, Process, and Visualize NGR Growth Data
        ###################################
        # Import MMA Data
        start_time = time.perf_counter()
        mma_monthly_growth_data_raw = query_snowflake(sf_queries['mma_growth_query'],session)
        mma_first_month_growth_data = query_snowflake(sf_queries['mma_firstmonth_query'],session)
        # st.write(mma_monthly_growth_data_raw.shape)
        
        
        # Import MRA Data
        mra_monthly_growth_data_raw = query_snowflake(sf_queries['mra_growth_query'],session)
        mra_first_month_growth_data = query_snowflake(sf_queries['mra_firstmonth_query'],session)
        end_time = time.perf_counter()
        duration = round(end_time - start_time,3)
        if print_times:
            st.write(f"Querying the monthly data took {duration} seconds to run")


        # Filter based on selected Classrooms and Schools
        def filter_orgs(df,slider_values):
            """
            Filters the given dataframe based on school, classroom, and grades
            """
            filter_criteria = {
                'SCHOOL_NAME': slider_values['selected_schools'],
                'CLASSROOM_NAME': slider_values['selected_classes'],
                'STUDENT_GRADE': slider_values['selected_grades']
            }
            for column, selected_values in filter_criteria.items():
                df = df[df[column].isin(selected_values)]
            return df
            
        mma_monthly_growth_data_raw = filter_orgs(mma_monthly_growth_data_raw,slider_values)
        mra_monthly_growth_data_raw = filter_orgs(mra_monthly_growth_data_raw,slider_values)

        # Process Data and Visualize it
        mma_first_last = process_monthly_growth_data(mma_monthly_growth_data_raw,mma_first_month_growth_data,product = 'My Math Academy',slider_values=slider_values)
        mra_first_last = process_monthly_growth_data(mra_monthly_growth_data_raw,mra_first_month_growth_data,product = 'My Reading Academy',slider_values=slider_values)
        st.title('Implementation Overview')

        # csv = pd.concat([mma_first_last,mra_first_last]).to_csv(index = False)
        # st.download_button(
        #     label="Download data as CSV",
        #     data=csv,
        #     file_name='data.csv',
        #     mime='text/csv',
        # )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.title("My Math Academy")
            # create_funnel_viz(mma_monthly_growth_data_raw, product='My Math Academy')
            # st.write(f"Final sample includes {mma_first_last['STUDENT_ID'].nunique()} total students.")
            st.write(aggregated_table(mma_first_last,['SCHOOL_NAME']))
            st.write(aggregated_table(mma_first_last,['STUDENT_GRADE']))
            st.write(aggregated_table(mma_first_last,['CLASSROOM_NAME']))
        # create_dumbbell_plot(mma_first_last,groupby = 'SCHOOL_NAME')
            # st.write(mma_first_last.groupby('STUDENT_GRADE')['STUDENT_ID'].nunique().reset_index(name = "Unique Student Count"))
        
        with col2:
            st.title("My Reading Academy")
            # create_funnel_viz(mra_monthly_growth_data_raw, product = 'My Reading Academy')
            # st.write(f"Final sample includes {mra_first_last['STUDENT_ID'].nunique()} total students.")
            st.write(aggregated_table(mra_first_last,['SCHOOL_NAME']))
            st.write(aggregated_table(mra_first_last,['STUDENT_GRADE']))
            st.write(aggregated_table(mra_first_last,['CLASSROOM_NAME']))
            # st.write(mra_first_last.groupby('STUDENT_GRADE')['STUDENT_ID'].nunique().reset_index(name = "Unique Student Count"))
        
        st.title('Are Students Progressing Toward Next Grade Readiness?')
        with st.expander("Definitions"):
            st.markdown("""
                <b><u>Next Grade Readiness</u></b> measures the percent of prior and current grade skills covered in My Math Academy/My Reading Academy of which the student has demonstrated knowledge.

                <b><u>Next Grade Ready:</u></b> Mastered at least 90% of At/Below grade level skills available in program.

                <b><u>On Track:</u></b> Mastered at least the percentage of skills needed for the current month to finish by end of year (~11% per month).

                <b><u>Caught Up:</u></b> The student was not On Track at their placement but now they are On Track.

                <b><u>Narrowed the Gap:</u></b> The student is closer to being On Track than they were at placement.

                <b><u>Below Target:</u></b> All other students not matching categories above.
                """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_kids_on_track(mma_first_last,size_font=slider_fontsize+4,product='My Math Academy')
        with col2:
            print('block')
            plot_growth_status(mma_first_last,slider_fontsize=slider_fontsize,product='My Math Academy')
            # plot_kids_on_track(mma_first_last, split_by='STUDENT_GRADE', size_font=slider_fontsize,product='My Math Academy')
        
        with col3:
            plot_kids_on_track(mra_first_last,size_font=slider_fontsize+4,product='My Reading Academy')
        with col4:
            print('block')
            plot_growth_status(mra_first_last, slider_fontsize=slider_fontsize,product='My Reading Academy')
            # plot_kids_on_track(mra_first_last, split_by='STUDENT_GRADE', size_font=slider_fontsize,product='My Reading Academy')

        # st.title("Top Students")
        # st.title('Are students catching up?')
        def display_top_sorted_union(df1, df2):
            rename_dict = {
                'STUDENT_ID': 'STUDENT_ID',
                'AOFL_PRODUCT': 'Product',
                'STUDENT_GRADE': 'Grade',
                'SCHOOL_NAME': 'School',
                'WEEKLY_USAGE_MIN': 'Weekly Usage Minutes',
                'ACTIVE_WEEKS': 'Active Weeks',
                'FIRST_NGR_BYPASSED_PCT': 'NGR - Placement',
                'CURRENT_GRADE_READINESS': 'NGR - Current',
                'MAX_GRADE_READINESS': 'NGR - Target',
                'NGR_GROWTH': '% Skills Mastered',
                # 'FIRST_MAX_GRADE_READINESS':'NGR - Target Placement',
                # 'FIRST_GAP':'FIRST_GAP',
                # 'CURRENT_GAP':'CURRENT_GAP'
            }
        
            format_dict = {
                'WEEKLY_USAGE_MIN': '{:.0f}',
                'ACTIVE_WEEKS': '{:.0f}',
                'FIRST_CURRENT_GRADE_READINESS': '{:.0%}',
                'CURRENT_GRADE_READINESS': '{:.0%}',
                'MAX_GRADE_READINESS': '{:.0%}',
                'NGR_GROWTH': '{:.0%}'
            }
        
            # Filter, sort by 'CURRENT_GRADE_READINESS', and select the top 20 rows
            top10_df1 = df1[list(rename_dict.keys())].sort_values(by='NGR_GROWTH', ascending=False).head(10)
            top10_df2 = df2[list(rename_dict.keys())].sort_values(by='NGR_GROWTH', ascending=False).head(10)
            
            # Concatenate the top 20 rows from each DataFrame
            combined_top10 = pd.concat([top10_df1, top10_df2])
            
            # Rename columns as specified in rename_dict
            combined_top10_renamed = combined_top10.rename(columns=rename_dict)
            
            # Apply formatting as the last step
            for column, original_name in rename_dict.items():
                if original_name in format_dict:
                    combined_top10_renamed[original_name] = combined_top10_renamed[original_name].map(format_dict[original_name].format)
        
            # Display the combined and renamed DataFrame in Streamlit
            st.table(combined_top10_renamed)
        # def display_top_sorted_union(df1, df2):
        #     # st.write(df1.columns)

        #     # Apply formatting to the original DataFrames using format_dict
        #     df1_formatted = df1.style.format(format_dict)
        #     df2_formatted = df2.style.format(format_dict)
        #     st.write(df1_formatted)
            
        #     # Convert back to DataFrame to perform sorting and filtering
        #     df1 = df1_formatted.data
        #     df2 = df2_formatted.data
        
        #     # Filter columns based on rename_dict keys and sort by 'CURRENT_GRADE_READINESS', then select the top 10 rows
        #     top10_df1 = df1[list(rename_dict.keys())].sort_values(by='CURRENT_GRADE_READINESS', ascending=False).head(10)
        #     top10_df2 = df2[list(rename_dict.keys())].sort_values(by='CURRENT_GRADE_READINESS', ascending=False).head(10)
            
        #     # Concatenate the top 10 rows from each DataFrame
        #     combined_top10 = pd.concat([top10_df1, top10_df2])
        
        #     # Rename columns as specified in rename_dict
        #     combined_top10_renamed = combined_top10.rename(columns=rename_dict)
            
        #     # Display the combined and renamed DataFrame in Streamlit
        #     st.dataframe(combined_top10_renamed)
        # display_top_sorted_union(mma_first_last,mra_first_last)
        
        
        #############################################
        # Query, Process, and Visualize Domain data
        #############################################
        if True:
            mma_domain_growth_raw = query_snowflake(sf_queries['mma_domain_query'],session)
            mra_domain_growth_raw = query_snowflake(sf_queries['mra_domain_query'],session)

            # Filter based on selected Classrooms and Schools
            mma_domain_growth_raw = filter_orgs(mma_domain_growth_raw,slider_values)
            mra_domain_growth_raw = filter_orgs(mra_domain_growth_raw,slider_values)

            # Create the domain dataset, containing the same sample as the monthly growth data, and preparing for visualization
            mma_student_ids = mma_first_last['STUDENT_ID'].unique()
            mma_domain_growth_sample = mma_domain_growth_raw[mma_domain_growth_raw['STUDENT_ID'].isin(mma_student_ids)]
            mma_domain_student_agg = process_skill_data(mma_domain_growth_sample)
            
            mra_student_ids = mra_first_last['STUDENT_ID'].unique()
            mra_domain_growth_sample = mra_domain_growth_raw[mra_domain_growth_raw['STUDENT_ID'].isin(mra_student_ids)]
            mra_domain_student_agg = process_skill_data(mra_domain_growth_sample)
            
            # st.title('What skills are students learning?')
            col1, col2 = st.columns(2)
            with col1:
                plot_skill_domain_distribution(mma_domain_student_agg,AOFL_PRODUCT='My Math Academy',sizefont=slider_fontsize)
            with col2:
                plot_skill_domain_distribution(mra_domain_student_agg,AOFL_PRODUCT='My Reading Academy',sizefont=slider_fontsize)
            #     plot_skill_domain_distribution(mma_domain_student_agg, split_by='STUDENT_GRADE',AOFL_PRODUCT='My Math Academy',sizefont = slider_fontsize)
            # with col3:
            #     plot_skill_domain_distribution(mra_domain_student_agg,AOFL_PRODUCT='My Reading Academy',sizefont=slider_fontsize)
            # with col4:
            #     plot_skill_domain_distribution(mra_domain_student_agg, split_by='STUDENT_GRADE',AOFL_PRODUCT='My Reading Academy',sizefont=slider_fontsize)

            st.title('Progress by Grade')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                plot_kids_on_track(mma_first_last,split_by='STUDENT_GRADE',size_font=slider_fontsize,product='My Math Academy')
            with col2:
                plot_skill_domain_distribution(mma_domain_student_agg, split_by='STUDENT_GRADE',AOFL_PRODUCT='My Math Academy',sizefont = slider_fontsize)
            with col3:
                plot_kids_on_track(mra_first_last,split_by='STUDENT_GRADE',size_font=slider_fontsize,product='My Reading Academy')
            with col4:
                plot_skill_domain_distribution(mra_domain_student_agg, split_by='STUDENT_GRADE',AOFL_PRODUCT='My Reading Academy',sizefont=slider_fontsize)
        

        # st.title('How did students perform, by grade, and by usage bucket?')
        # render_mpl_table(pd.concat([mma_first_last, mra_first_last]))
        

        session.close()

if __name__ == '__main__':
    main()
    