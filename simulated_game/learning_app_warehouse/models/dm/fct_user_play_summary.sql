{{ config(materialized='table') }}

with base as (
select 
    play.user_id,
    dus.location as user_state,
    dus.age as user_age,
    dus.gender as user_gender,
    dus.sample_id as user_sample_id,
    dus.created_at as user_created_at,
    count(distinct level_attempt_id) as total_play_attempts,
    count(distinct case when completed = true and passed = true then level_attempt_id end) as total_completed_attempts,
    count(distinct case when completed = true and passed = false then level_attempt_id end) as total_failed_attempts,
    count(distinct case when completed = false then level_attempt_id end) as total_canceled_attempts,
    count(distinct case when completed = true and passed = false then level end) as total_failed_levels,
    sum(session_duration_minutes) as total_session_duration_minutes,
    sum(case when completed = true then session_duration_minutes end) as total_completed_duration_minutes,
    sum(case when completed = false then session_duration_minutes end) as total_canceled_duration_minutes,
    avg(case when completed = true then session_duration_minutes end) as avg_completed_duration_minutes,
    avg(case when completed = false then session_duration_minutes end) as avg_canceled_duration_minutes,
    max(case when completed = true and passed = true then level end) as max_level_completed,
    max(case when completed = true and passed = false then level end) as max_level_failed,
    count(distinct start_ts::date) as total_days_played,
    min(start_ts) as first_play_ts,
    max(start_ts) as last_play_ts,
    count(distinct case when completed = true and passed = true and date_trunc('month', dus.created_at) = date_trunc('month', start_ts) then level end) as total_completed_levels_first_month,
    datediff('month',min(start_ts), max(start_ts)) + 1 as total_months_since_first_play
from {{ ref('fct_play_attempts')}} play
left join {{ ref('dim_user')}} dus on play.user_id = dus.user_id
group by play.user_id, dus.location, dus.age, dus.gender, dus.sample_id, dus.created_at
)

select * from base