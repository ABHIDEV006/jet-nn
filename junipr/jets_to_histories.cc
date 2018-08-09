// compile:
// g++ jets_to_histories.cc -o jets_to_histories `../install/bin/fastjet-config --cxxflags --libs --plugins`
// run:
// ./jets_to_histories input_directory input_file output_directory recluster_def

#include "fastjet/ClusterSequence.hh"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace fastjet;
using namespace std;

#define NEVENTS    1000000

class Charge: public PseudoJet::UserInfoBase {
  protected:
    double charge;
  public:
    Charge(double c) {
      charge = c;
    }

    double get_charge() const {
      return charge;
    }
};

int decluster(PseudoJet j, ofstream& file, int label, int count, double charges[]);
int charge(PseudoJet j, int label, int count, double * charges);
int num_tree_nodes(PseudoJet j);


int main(int argc, char *argv[]) {

  if (argc != 5) {    
    cout << "Wrong number of arguments" << endl;
    return 0;
  }
  
  // User input
  string input_directory = argv[1];
  string input_file = argv[2];
  string output_directory = argv[3];
  double recluster_def = atof(argv[4]);

  // Input, output
  stringstream s_infile;
  s_infile << input_directory << "/" << input_file;
  ifstream infile;
  infile.open( s_infile.str().c_str() );

  stringstream s_outfile;
  s_outfile << output_directory << "/reclustered_" << input_file ;
  ofstream outfile;
  outfile.open( s_outfile.str().c_str());
  

  // Read jets from input file
  // Jets begin with J <jet number> N <number of subjets>
  // Following N lines are px py pz e q for subjets
  string dummy;
  int ijet;
  int nsubjets;

  // Loop over jets
  while ((infile >> dummy) && (infile >> ijet) && (ijet <= NEVENTS)) {

    if (ijet % 10000 == 0) cout << "processing jet " << ijet << endl;

    infile >> dummy >> nsubjets;

    vector<PseudoJet> particles;

    // Particle loop
    for (int i = 0; i < nsubjets; i++) {

      double px, py, pz, e, q;
      infile >> px >> py >> pz >> e >> q;

      PseudoJet particle(px, py, pz, e);
      particle.set_user_info(new Charge(q));
      particles.push_back(particle);
      
    }  // end loop (particles)
    
    // Recluster jet constituents
    // change to genkt_algorithm, recluster_def -> 0 for purpose of ud jets
    JetDefinition reclust_def(genkt_algorithm, 1.5708, recluster_def); // radius unused
    ClusterSequence reclust_seq(particles, reclust_def);
    vector<PseudoJet> reclust_jets = reclust_seq.exclusive_jets(1);
    if (ijet == 0)
      cout << "Reclustered with " << reclust_def.description() << endl;
    
    PseudoJet tree;
    tree.reset( reclust_jets[0] );
    int length = num_tree_nodes(tree);
    double charges[length];
    for (int i = 0; i < length; ++i) {
      charges[i] = 0;
    }
    charge(tree, 0, 0, charges);

    outfile << "J " << tree.px() << " " << tree.py() << " " << tree.pz() << " " << tree.e() << " " << charges[0] << endl;
    outfile << "S " << nsubjets - 1 << endl;
    
    decluster(tree, outfile, 0, 0, charges);

  } // end loop (jets)
  
  infile.close();
  outfile.close();
  return 0;

} // end function (main)

int num_tree_nodes(PseudoJet j) {
  PseudoJet j1, j2;
  if (j.has_parents(j1, j2))
    return num_tree_nodes(j1) + num_tree_nodes(j2) + 1;
  else
    return 1;
} // end function (num_tree_nodes)

int charge(PseudoJet j, int label, int count, double * charges) {

  PseudoJet j1, j2;

  if (j.has_parents(j1, j2)) {
    PseudoJet jh, js;
    if (j1.e() > j2.e()) {
        jh = j1;
        js = j2;
    }
    else {
        jh = j2;
        js = j1;
    }
    int updated_count = count + 2;
    updated_count = charge(jh, count+1, updated_count, charges);
    updated_count = charge(js, count+2, updated_count, charges);
    charges[label] = charges[count + 1] + charges[count + 2];
    return updated_count;
  } else {
    charges[label] = j.user_info<Charge>().get_charge();
    return count;
  }

} // end function (charge)

int decluster(PseudoJet j, ofstream& file, int label, int count, double charges[]) {

  PseudoJet j1, j2;

  if (!j.has_parents(j1,j2)) return count;
  
  // NEW CODE
  PseudoJet jh, js;
  if (j1.e() > j2.e()) {
      jh = j1;
      js = j2;
  }
  else {
      jh = j2;
      js = j1;
  }

  file << label << " ";
  file << jh.px() << " " << jh.py() << " " << jh.pz() << " " << jh.e() << " " << charges[count + 1] << " ";
  file << js.px() << " " << js.py() << " " << js.pz() << " " << js.e() << " " << charges[count + 2] << endl;

  int updated_count = count + 2;
  updated_count = decluster(jh, file, count+1, updated_count, charges);
  updated_count = decluster(js, file, count+2, updated_count, charges);
  
  return updated_count;

} // end function (decluster)


