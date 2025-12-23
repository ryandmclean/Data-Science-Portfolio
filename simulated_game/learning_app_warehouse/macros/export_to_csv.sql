{% macro export_model_to_csv(model_name, output_path) %}
  {% set relation = ref(model_name) %}

  {% set sql %}
    copy {{ relation }}
    to '{{ output_path }}'
    (header, delimiter ',');
  {% endset %}

  {% do run_query(sql) %}
{% endmacro %}
