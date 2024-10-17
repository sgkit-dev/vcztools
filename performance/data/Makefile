# The make recipes require bcftools and bgzip.

# https://samtools.github.io/bcftools/howtos/install.html
# https://www.htslib.org/doc/bgzip.html

# On macOS, there are Homebrew formulas for bcftools and htslib,
# which contains bgzip.

# The Python requirements are listed in requirements.txt:
# pip install -r requirements.txt

# Flags / commandline arguments:
CHROMOSOME ?= 22
WGS ?= 1

ifeq ($(WGS), 1)
        TGP_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20190425_NYGC_GATK/CCDG_13607_B01_GRM_WGS_2019-02-19_chr$(CHROMOSOME).recalibrated_variants.vcf.gz"
else
        # Use URL for genotyping data:
        TGP_URL = "http://hgdownload.cse.ucsc.edu/gbdb/hg19/1000Genomes/phase3/ALL.chr$(CHROMOSOME).phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
endif

.PHONY: all simulated real clean

all: simulated real

simulated: sim_10k.vcz

real: chr22.vcz

sim_10k.ts:
	stdpopsim HomSap -c chr22 -o sim_10k.ts pop_0:10000

chr22.vcf.gz:
	bcftools view $(TGP_URL) | head -n 25000 | bcftools view -O z -o chr22.vcf.gz

%.vcf.gz: %.ts
	tskit vcf $< | bgzip > $@

%.vcf.gz.csi: %.vcf.gz
	bcftools index $<

%.vcz: %.vcf.gz %.vcf.gz.csi
	vcf2zarr convert $< $@

clean:
	rm -rf sim_10k.*
