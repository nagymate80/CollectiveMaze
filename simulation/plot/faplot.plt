#!/usr/bin/gnuplot

set title ""


set size ratio -1
set yrange [0.4:1.05] #reverse
set xrange [0.05:0.95]

set format x ""
set format y ""
set xtics (0)
set ytics (0)

cr=0.04
transp=1.0

set palette model HSV defined ( 0 0 1 1, 1 0.9 1 1 )

file1="tree4.txt"
#do for [l=0:200]{

file2=sprintf("../out0.txt1")

#network 1st row, 2nd row nodes

plot file1 using 1:2:3:4 notitle w vector nohead linewidth 6 lt rgb "black"

#     file2  using 1:2:(cr):3:4:5 notitle  w circles lc palette fs transparent solid transp noborder


pause -1 "Press enter!"

set term jpeg
set out sprintf("plot_%s.jpg",file2)

replot
#}
