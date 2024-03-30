./perlbench_base.amd64-m64-gcc42-nn -I. -I./lib attrs.pl
./bzip2_base.amd64-m64-gcc42-nn input.program 5
./gcc_base.amd64-m64-gcc42-nn cccp.i -o cccp.s
./bwaves_base.amd64-m64-gcc42-nn
./gromacs_base.amd64-m64-gcc42-nn -silent -deffnm gromacs -nice 0
./mcf_base.amd64-m64-gcc42-nn inp.in
./milc_base.amd64-m64-gcc42-nn < su3imp.in
./zeusmp_base.amd64-m64-gcc42-nn
./gromacs_base.amd64-m64-gcc42-nn -silent -deffnm gromacs -nice 0
./cactusADM_base.amd64-m64-gcc42-nn benchADM.par
./leslie3d_base.amd64-m64-gcc42-nn < leslie3d.in
./namd_base.amd64-m64-gcc42-nn --input namd.input --iterations 1 --output namd.out
./gobmk_base.amd64-m64-gcc42-nn --quiet --mode gtp < capture.tst
./soplex_base.amd64-m64-gcc42-nn -m10000 test.mps
./povray_base.amd64-m64-gcc42-nn SPEC-benchmark-test.ini
./calculix_base.amd64-m64-gcc42-nn -i  beampic
./hmmer_base.amd64-m64-gcc42-nn --fixed 0 --mean 325 --num 45000 --sd 200 --seed 0 bombesin.hmm
./sjeng_base.amd64-m64-gcc42-nn test.txt
./GemsFDTD_base.amd64-m64-gcc42-nn
./libquantum_base.amd64-m64-gcc42-nn 33 5
./h264ref_base.amd64-m64-gcc42-nn -d foreman_test_encoder_baseline.cfg
./tonto_base.amd64-m64-gcc42-nn
./lbm_base.amd64-m64-gcc42-nn 20 reference.dat 0 1 100_100_130_cf_a.of
./omnetpp_base.amd64-m64-gcc42-nn omnetpp.ini
./astar_base.amd64-m64-gcc42-nn lake.cfg
./wrf_base.amd64-m64-gcc42-nn
./sphinx_livepretend_base.amd64-m64-gcc42-nn ctlfile . args.an4
./specrand_base.amd64-m64-gcc42-nn 324342 24239
