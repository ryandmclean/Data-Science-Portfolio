{{ config(materialized='table') }}

select 
    user_id,
    payload ->> 'location' as location,
    payload ->> 'age' as age,
    payload ->> 'gender' as gender,
    payload ->> 'sample_id' as sample_id,
    timestamp as created_at,
from {{ ref('event_stream') }}
where event_type = 'user_created'