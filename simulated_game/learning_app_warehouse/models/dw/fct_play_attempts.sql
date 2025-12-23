{{ config(materialized='table') }}

select
    event_id,
    user_id,
    timestamp as start_ts,
    timestamp + ((payload ->> 'session_duration_seconds')::int * interval '1 second') as end_ts,
    payload ->> 'level_attempt_id' as level_attempt_id,
    (payload ->> 'level')::int as level,
    payload ->> 'device' as device,
    (payload ->> 'completed')::boolean as completed,
    (payload ->> 'passed')::boolean as passed,
    (payload ->> 'session_duration_seconds')::int as session_duration_seconds,
    session_duration_seconds::int / 60 as session_duration_minutes
from {{ ref('event_stream') }}
where event_type = 'play_attempt'