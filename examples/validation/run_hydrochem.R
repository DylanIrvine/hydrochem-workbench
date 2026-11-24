# run_hydrochem.R
# runs the Quebec dataset that ships with the R package hydrochem (Cinkus and Bondu, 2026,
# Groundwater, doi:10.1111/gwat.70110) through that package, and writes two files for
# compare_with_hydrochem.py:
#   data_qc_raw.csv         the dataset as shipped (146 samples, censored values as "< x")
#   hydrochem_results.csv   what hydrochem calculates from it
#
# once, in R:  install.packages("hydrochem")
# then:        Rscript run_hydrochem.R     (or run it in RStudio, with this folder as the working directory)

library(hydrochem)
library(data.table)

data("data_qc", package = "hydrochem")
raw <- as.data.table(data_qc)
fwrite(raw, "data_qc_raw.csv")

# censored values at half the detection limit, the same rule as the Workbench default
d <- replace_bdl(raw, method = "DL/2")

out <- data.table(Sample = d$Sample_ID)
out$ib_major       <- ionic_balance(d, base_unit = "mg/L", method = "major")
out$ib_major_minor <- ionic_balance(d, base_unit = "mg/L", method = "major_minor")
out$tds_major      <- tds(d, base_unit = "mg/L", method = "major")
out$tds_all        <- tds(d, base_unit = "mg/L", method = "all_ions")
out$hardness       <- total_hardness(d, base_unit = "mg/L", output = "mg/L_CaCO3")
out$alkalinity     <- total_alkalinity(d, base_unit = "mg/L", output = "mg/L_CaCO3")
out$sar            <- sar(d, base_unit = "mg/L")
out$percent_sodium <- percent_sodium(d, base_unit = "mg/L")
out$cai1           <- cai1(d, base_unit = "mg/L")
out$water_type     <- water_type(d, base_unit = "mg/L", format = "short")
fwrite(out, "hydrochem_results.csv")

cat("hydrochem", as.character(packageVersion("hydrochem")), ":", nrow(out), "samples written\n")
