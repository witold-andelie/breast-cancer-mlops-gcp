-- NAME: vw_dataset
CREATE OR REPLACE VIEW vw_dataset AS
SELECT *,
       CASE WHEN {{target_col}} = '{{positive_label}}' THEN 1 ELSE 0 END AS y_m,
       CASE WHEN CAST(hash({{id_col}}) AS UBIGINT) % {{hash_mod}} < {{train_pct}}
            THEN 'train' ELSE 'test' END AS partition
FROM src_clean;

-- NAME: partition_counts
SELECT partition,
       {{target_col}} AS diagnosis,
       COUNT(*) AS n
FROM vw_dataset
GROUP BY partition, {{target_col}}
ORDER BY partition, {{target_col}};
