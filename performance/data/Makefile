# The make recipes require bcftools and bgzip.

# https://samtools.github.io/bcftools/howtos/install.html
# https://www.htslib.org/doc/bgzip.html

# On macOS, there are Homebrew formulas for bcftools and htslib,
# which contains bgzip.

# The Python requirements are listed in requirements.txt:
# pip install -r requirements.txt

.PHONY: all clean

all: sim_10k.vcz

sim_10k.ts:
	stdpopsim HomSap -c chr22 -o sim_10k.ts pop_0:10000

%.vcf.gz: %.ts
	tskit vcf $< | bgzip > $@

%.vcf.gz.csi: %.vcf.gz
	bcftools index $<

%.vcz: %.vcf.gz %.vcf.gz.csi
	vcf2zarr convert $< $@

clean:
	rm -rf sim_10k.*
