// Christopher Frye
// 16 June 2017
// Cluster final state into jets, find small subjets, print 4-momenta

// compile: g++ events_to_jets.cc -o events_to_jets `../install/bin/fastjet-config --cxxflags --libs --plugins`
// run: ./events_to_jets subjet_emin subjet_radius input_file output_prefix recluster_def



#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace fastjet;
using namespace std;

#define NEVENTS     500000
#define DEBUG       0



int main(int argc, char *argv[]) {

  if (argc != 6) {    
    cout << "Wrong number of arguments" << endl;
    return 0;
  }
  
  // User input
  double subjet_emin = atof(argv[1]);
  double subjet_radius = atof(argv[2]);
  string input_str = argv[3];
  string output_prefix = argv[4];
  double recluster_def = atof(argv[5]); // 0 for C/A, 1 for kt, -1 for anti-kt

  stringstream output_str;
  output_str << output_prefix << "_j" << subjet_emin << "_r" << subjet_radius << "_p" << recluster_def << ".dat" ;

  ifstream infile;
  infile.open( input_str.c_str() );
  ofstream outfile;
  outfile.open( output_str.str().c_str() );

  // Read events from input file
  // Lines begin with E (event) or P (particle)
  // Following entry is event # or particle #
  string linetype;
  int ievent;
  string dummy;

  int num_events = 0;
  int num_printed_jets = 0;

  // Event loop
  infile >> linetype;
  while (infile >> ievent) {

    num_events++;
    if (num_events > NEVENTS) break;

    if (num_events % 10000 == 0) cout << "processing event " << num_events << endl;

    vector<PseudoJet> particles;

    // Particle loop
    while ((infile >> linetype) && (linetype == "P")) {

      // read particle number
      infile >> dummy;

      // read particle info
      int id;
      double px, py, pz, e;
      infile >> id >> px >> py >> pz >> e;

      // what not to pass to fastjet:

      // no neutrinos
      if (abs(id) == 12 || abs(id) == 14 || abs(id) == 16) continue;

      // NO CUT ON PSEUDORAPIDITY
      // double p = sqrt(px*px + py*py + pz*pz);
      // double pseudorap = 0.5 * log( (p+pz)/(p-pz) );
      // if (abs(pseudorap) > ETAMAX) continue;

      // give remaining particle to fastjet, store id
      PseudoJet part(px, py, pz, e);
      part.set_user_index(id);
      particles.push_back(part);
      
    }  // end loop (particles)
    


    // Cluster final-state particles into hemispheres using the exclusive kt algorithm
    JetDefinition jet_def(ee_kt_algorithm);
    ClusterSequence clust_seq(particles, jet_def);
    vector<PseudoJet> jets = clust_seq.exclusive_jets(2);
    if (num_events == 1)
      cout << "Defined hemispheres using exclusive jets from: " << jet_def.description() << endl;

    // Cluster jets' constituents into subjets with user-defined clustering algorithm
    int njets = 2;
    for (int j = 0; j < njets; j++) {

      // Cluster jet into subjets
      JetDefinition subjet_def(ee_genkt_algorithm, subjet_radius, recluster_def);
      vector<PseudoJet> subjets = subjet_def(jets[j].constituents());
      if (num_events == 1 && j == 0) 
	cout << "Reclustered hemispheres with " << subjet_def.description() << endl;

      // Use all subjets with E > subjet_emin
      int nsubjets = 0;
      for (int i = 0; i < subjets.size(); i++) {
	
	if (subjets[i].e() < subjet_emin) break;
	nsubjets++;
      }
      if (nsubjets == 0) continue;
      
      // Print subjet info
      outfile << "J " << num_printed_jets << endl;
      num_printed_jets++;
      outfile << "N " << nsubjets << endl;
      
      for (int sj = 0; sj < nsubjets; sj++) {

	outfile << subjets[sj].px() << " " ;
	outfile << subjets[sj].py() << " " ;
	outfile << subjets[sj].pz() << " " ;
	outfile << subjets[sj].e() << endl;
      }

    } // end loop (jets)
    
  } // end loop (events)



  infile.close();
  outfile.close();
  return 0;

} // end function (main)
