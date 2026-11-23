-- NAME: class_distribution
SELECT {{target_col}} AS diagnosis,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM src
GROUP BY {{target_col}}
ORDER BY {{target_col}};

-- NAME: missing_summary
SELECT column_name,
       COUNT(*) - COUNT(val) AS n_null
FROM (
    SELECT * FROM src
    UNPIVOT (val FOR column_name IN ({{numeric_cols}}))
)
GROUP BY column_name
ORDER BY column_name;

-- NAME: numeric_profile
SELECT column_name,
       ROUND(MIN(val), 4) AS min_v,
       ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY val), 4) AS median_v,
       ROUND(MAX(val), 4) AS max_v,
       ROUND(AVG(val), 4) AS mean_v,
       ROUND(STDDEV_SAMP(val), 4) AS sd,
       ROUND(SKEWNESS(val), 4) AS skew
FROM (
    SELECT * FROM src
    UNPIVOT (val FOR column_name IN ({{numeric_cols}}))
)
GROUP BY column_name
ORDER BY column_name;

-- NAME: iqr_outliers
WITH quartiles AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY area_mean)    AS q1_area_mean,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY area_mean)    AS q3_area_mean,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY area_se)      AS q1_area_se,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY area_se)      AS q3_area_se,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY texture_mean) AS q1_texture,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY texture_mean) AS q3_texture
    FROM src
)
SELECT
    SUM(CASE WHEN area_mean < q1_area_mean - 1.5 * (q3_area_mean - q1_area_mean)
              OR area_mean > q3_area_mean + 1.5 * (q3_area_mean - q1_area_mean)
             THEN 1 ELSE 0 END) AS n_out_area_mean,
    SUM(CASE WHEN area_se < q1_area_se - 1.5 * (q3_area_se - q1_area_se)
              OR area_se > q3_area_se + 1.5 * (q3_area_se - q1_area_se)
             THEN 1 ELSE 0 END) AS n_out_area_se,
    SUM(CASE WHEN texture_mean < q1_texture - 1.5 * (q3_texture - q1_texture)
              OR texture_mean > q3_texture + 1.5 * (q3_texture - q1_texture)
             THEN 1 ELSE 0 END) AS n_out_texture_mean
FROM src, quartiles;
