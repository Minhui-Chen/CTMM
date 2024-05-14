library("countsimQC")

output_dir <- dirname(snakemake@output[['html']])
html <- basename(snakemake@output[['html']])
rds <- paste0(tools::file_path_sans_ext(html), "_ggplots.rds")

# make dds
raw <- as.matrix(read.table(snakemake@input[['raw']], header=T, row.names='gene'))
sim <- as.matrix(read.table(snakemake@input[['sim']], header=T, row.names='gene'))
 
data <- list(real=raw, simulated=sim)

# remove existing rmd file 
rmd <- paste0(tools::file_path_sans_ext(html), '.Rmd')
if (file.exists(rmd)) {
    file.remove(rmd)
}

countsimQCReport(ddsList=data, outputFile=html, savePlots=T, outputDir='./', 
                    calculateStatistics=T, permutationPvalues=F)
file.rename(html, paste0(output_dir, '/', html))

# generate individual figures
ggplots <- readRDS(rds)

if (!dir.exists(file.path(output_dir, "figures"))) {
    dir.create(file.path(output_dir, "figures"))
}

generateIndividualPlots(ggplots, device='png', nDatasets=2, 
                        outputDir=file.path(output_dir, "figures"))

file.remove(rds) 