import polars as pl

lf = pl.scan_csv("token_transfers_V2.0.0.csv")

lf_from = lf.select([
    pl.col("from_address").alias("address"),
    (pl.col("time_stamp") * 2).alias("val")
])

lf_to = lf.select([
    pl.col("to_address").alias("address"),
    (pl.col("time_stamp") * 2 + 1).alias("val")
])

result_df = (
    pl.concat([lf_from, lf_to])
    .group_by("address")
    .agg(pl.col("val").unique())
    .collect(streaming=True)
)

result_df.write_parquet("transfers.parquet")
