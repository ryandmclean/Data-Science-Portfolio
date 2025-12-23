{{ config(materialized='table') }}

with base as (
    select 
        user_id,
        start_ts,
        device,
        event_id
    from {{ ref('fct_play_attempts')}}
    where device is not null 
), 

ordered as (
    select 
        *,
        lag(device) over (partition by user_id order by start_ts) as prev_device
    from base
),

change_points as (
    select 
        user_id,
        device,
        cast(start_ts as timestamp) as valid_from,
        row_number() over (partition by user_id order by start_ts) as device_change_id
    from ordered
    where device != prev_device or prev_device is null
),

final as (
    select 
    user_id,
    device,
    device_change_id,
    valid_from,
    coalesce(
        lead(valid_from) over (partition by user_id order by valid_from),
        cast('{{ var("future_timestamp") }}' as timestamp)
    ) as valid_to
    from change_points
)

select * from final