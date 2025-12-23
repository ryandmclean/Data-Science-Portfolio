{{ config(materialized='table') }}

select
  event_id,
  user_id,
  cast(timestamp as timestamp) as timestamp,
  event_type,
  payload
from read_csv_auto('./data/raw/events.csv', header=true)
