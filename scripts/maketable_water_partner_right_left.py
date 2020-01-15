
outfile = open('/home/attila/PATEKL/video/water_partner/water_partner_tables/water_partner_beta_gamma_toplot.txt', 'w')

infile = open('/home/attila/PATEKL/video/water_partner/water_partner_tables/water_partner_beta_gamma.txt', 'r')

line = ""

for node in range(3,17):
    for d in ['r', 'l']:
        outfile.write(str(node)+d+'\t')
        for i in range(4):
            line = infile.readline()
        outfile.write('\t'.join(line.split('\t')[1:3])+'\t')
        line = infile.readline()
        outfile.write('\t'.join(line.split('\t')[1:3])+'\n')
        for i in range(3):
            line = infile.readline()

infile.close()
outfile.close()
