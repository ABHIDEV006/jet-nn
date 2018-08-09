// Christopher Fye
// 16 June 2017
// Modified main06.cc to generate e+e- --> Z --> quarks for parton shower machine learning project
// Example call: ./parton_shower_learning LEP_91GeV_dijets.out 

#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

#include "Pythia8/Pythia.h"
using namespace Pythia8;

#include "fastjet/ClusterSequence.hh"
using namespace fastjet;

#define NJETS     1000000
#define MINENERGY 450.
#define MAXENERGY 550.
#define KTTYPE    -1.
#define ESUB      1.
#define RSUB      0.1

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

int main(int argc, char *argv[]) {
  
  if (argc < 3) {
    cout << "Too few arguments" << endl;
    return 0;
  }
  
  // User input
  string seed = argv[1];
  string file_str = argv[2];

  // File for output
  ofstream file; 
  file.open(file_str.c_str());

  // Generator.
  Pythia pythia;
  Event& event = pythia.event;

  /*
  pythia.settings.flag("HardQCD:qq2qq", true);
//  pythia.settings.flag("HardQCD:gg2qqbar", ggqb);
//  pythia.settings.flag("HardQCD:qqbar2gg", qbgg);
//  pythia.settings.flag("HardQCD:gg2gg", gggg);

  // Z decay settings
  pythia.readString("WeakZ0:gmZmode = 2");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 12 14 16");

  // Random seed
  pythia.settings.flag("Random:setSeed", true);
  pythia.readString("Random:seed = " + seed);

  // generation cuts and ptpow
  pythia.settings.parm("PhaseSpace:pTHatMin", pthatmin);
  pythia.settings.parm("PhaseSpace:pTHatMax", pthatmax);
//  pythia.settings.parm("PhaseSpace:bias2SelectionPow", ptpow);
//  pythia.settings.flag("PhaseSpace:bias2Selection", ptpow >= 0 ? true : false);

  // Initialisation
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 1000.");
  */
  
  
  // Random seed based on time
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + seed);

  // Display settings
  pythia.readString("Next:numberCount = 10000");
  pythia.readString("Next:numberShowEvent = 1");
  pythia.readString("Next:numberShowProcess = 1");
  
  // Collider settings
  pythia.readString("Beams:idA =  11");
  pythia.readString("Beams:idB = -11");
  //double mZ = pythia.particleData.m0(23);
  //pythia.settings.parm("Beams:eCM", mZ);
  pythia.readString("Beams:eCM = 1000.");
  pythia.readString("PDF:lepton = off");
  
  // Process selection
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 1 2 3 4 5");
  
  pythia.init();
  
  // Begin event loop. Generate event. Skip if error. List first few.
  int n_printed_jets = 0;
  bool already_printed = false;
  while (n_printed_jets < NJETS) {
    if (!pythia.next()) continue;

    if (n_printed_jets % 1000 == 0 && !already_printed) {
      cout << "printed jet " << n_printed_jets << endl;
      already_printed = true;
    }
    
    // Give observable final state particles to FastJet
    vector<PseudoJet> particles;
    for (unsigned int i = 0; i < event.size(); i++) {
      if (!event[i].isFinal()) continue;
	      
      // no neutrinos
      int abs_id = abs(event[i].id());
      if (abs_id == 12 || abs_id == 14 || abs_id == 16) continue;
      
      // give remaining particle to fastjet, store id
      PseudoJet particle(event[i].px(), event[i].py(), event[i].pz(), event[i].e());
      // just in case we might want to the particle id instead (or in addition
      // to) what we have
      // particle.set_user_index(event[i].id());
      particle.set_user_info(new Charge(event[i].charge()));
      particles.push_back(particle);
      
    }  // end loop (particles)

    // Cluster final-state particles into hemispheres using the exclusive kt algorithm
    JetDefinition jet_def(ee_kt_algorithm);
    ClusterSequence clust_seq(particles, jet_def);
    vector<PseudoJet> jets = clust_seq.exclusive_jets(2);

    // Only keep jets in mass window -- and energy window!
    int njets = 2;
    vector<PseudoJet> mass_jets;
    for (int j = 0; j < njets; j++) {
      if (jets[j].e() > MINENERGY && jets[j].e() < MAXENERGY)
	mass_jets.push_back(jets[j]);
    }
        
    // Cluster jets' constituents into subjets with user-defined clustering algorithm
    for (unsigned int j = 0; j < mass_jets.size(); j++) {
      
      // Cluster jet into subjets
      JetDefinition subjet_def(ee_genkt_algorithm, RSUB, KTTYPE);
      vector<PseudoJet> subjets = subjet_def(mass_jets[j].constituents());
      
      // Use all subjets with E > subjet_emin
      int nsubjets = 0;
      for (unsigned int i = 0; i < subjets.size(); i++) {
        if (subjets[i].e() < ESUB) break;
        nsubjets++;
      }
      if (nsubjets == 0) continue;
      
      // Print subjet info
      file << "J " << n_printed_jets << endl;
      n_printed_jets++;
      if (already_printed) already_printed = false;
      file << "N " << nsubjets << endl;
      
      for (int sj = 0; sj < nsubjets; sj++) {
        vector<PseudoJet> particles = subjets[sj].constituents();
        double charge = 0;
        for (int c = 0; c < particles.size(); ++c)
          charge += particles[c].user_info<Charge>().get_charge();
        file << subjets[sj].px() << " " ;
        file << subjets[sj].py() << " " ;
        file << subjets[sj].pz() << " " ;
        file << subjets[sj].e() << " ";
        file << charge << endl;
      }
      
    } // end loop (jets)
        
  } // end loop (events)
  
  pythia.stat();
  file.close();
  return 0;
}
