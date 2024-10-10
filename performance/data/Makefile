.PHONY: all clean

all: sim_10k.vcz

sim_10k.ts:
	stdpopsim HomSap -c chr22 -o sim_10k.ts pop_0:10000

sim_10k.vcf.gz: sim_10k.ts
	tskit vcf sim_10k.ts | bgzip > sim_10k.vcf.gz

sim_10k.vcf.gz.csi: sim_10k.vcf.gz
	bcftools index sim_10k.vcf.gz

sim_10k.vcz: sim_10k.vcf.gz sim_10k.vcf.gz.csi
	vcf2zarr convert sim_10k.vcf.gz sim_10k.vcz

clean:
	rm -rf sim_10k.*
