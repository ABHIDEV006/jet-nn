../bin/events_lh ../../data/junipr/d_jets.lhe 5 down
./bin/process_events_0.03 5 fewer_d_jets_0.03_ca.out ../../data/junipr/d_jets.lhe down
./bin/jets_to_histories . fewer_d_jets_0.03_ca.out . 0
python histories_to_final.py . reclustered_fewer_d_jets_0.03_ca.out ../../data/junipr
