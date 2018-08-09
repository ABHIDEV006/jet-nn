../bin/events_lh ../../data/junipr/u_jets.lhe 5 up
./bin/process_events_0.03 5 u_jets_0.03_ca.out ../../data/junipr/u_jets.lhe up
./bin/jets_to_histories . u_jets_0.03_ca.out . 0
python histories_to_final.py . reclustered_u_jets_0.03_ca.out ../../data/junipr

#./bin/process_events_0.03_no_cut 5 fewer_u_jets_0.03_no_cut.out ../../data/junipr/u_jets.lhe up
#./bin/jets_to_histories . fewer_u_jets_0.03_no_cut.out . -1
#python histories_to_final.py . reclustered_fewer_u_jets_0.03_no_cut.out ../../data/junipr
#
#./bin/process_events_no_cut 5 fewer_u_jets_no_cut.out ../../data/junipr/u_jets.lhe up
#./bin/jets_to_histories . fewer_u_jets_no_cut.out . -1
#python histories_to_final.py . reclustered_fewer_u_jets_no_cut.out ../../data/junipr
