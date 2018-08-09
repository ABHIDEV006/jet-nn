// Christopher Fye
// 16 June 2017
// Modified main06.cc to generate e+e- --> Z --> quarks for parton shower machine learning project
// Example call: ./parton_shower_learning LEP_91GeV_dijets.out 

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
using namespace std;

#include "Pythia8/Pythia.h"
using namespace Pythia8;

#include "fastjet/ClusterSequence.hh"
using namespace fastjet;

//#define NJETS     10000
#define NJETS     5000
#define MINENERGY 450.
#define MAXENERGY 550.
#define KTTYPE    -1.
#define ESUB      1.
#define RSUB      0.1
#define K         0.4

// defining a class for an event's daughter's list
class EventDL {
  protected:
    vector<int> mother1;
    vector<int> mother2;
    vector< vector<int> > daughters;
  public:
    void add(int m1, int m2, vector<int> dlist) {
      if (m1 < m2) {
        mother1.push_back(m1);
        mother2.push_back(m2);
      } else {
        mother1.push_back(m2);
        mother2.push_back(m1);
      }
      daughters.push_back(dlist);
    }

    vector<int> find(int d, Event event) {
      vector<int> ret;
      for (int i = 0; i < daughters.size(); ++i)
        for (int j = 0; j < daughters[i].size(); ++j)
          if (d == daughters[i][j]) {
            if (ret.empty()) {
              ret.push_back(mother1[i]);
              ret.push_back(mother2[i]);
            } else if (event[ret[0]].e() + event[ret[1]].e() < event[mother1[i]].e() + event[mother2[i]].e()) {
                ret.pop_back();
                ret.pop_back();
                ret.push_back(mother1[i]);
                ret.push_back(mother2[i]);
            }
          }
      return ret;
    }
};

class ID: public PseudoJet::UserInfoBase {
  protected:
    int id, index;
    double charge, color, acolor, pt, eta, phi;
  public:
    ID(int i, double c, double col, double acol, double p, int ind, double e, double ph) {
      id = i;
      charge = c;
      color = col;
      acolor = acol;
      pt = p;
      index = ind;
      eta = e;
      phi = ph;
    }

    int get_id() const {
      return id;
    }

    double get_charge() const {
      return charge;
    }

    double get_color() const {
      return color;
    }

    double get_acolor() const {
      return acolor;
    }

    double get_pt() const {
      return pt;
    }

    int get_index() const {
      return index;
    }

    double get_phi() const {
      return phi;
    }

    double get_eta() const {
      return eta;
    }
};

int main(int argc, char *argv[]) {
  
  string seed = argv[1];
  string file_str = argv[2];
  
  // File for output
  ofstream file; 
  file.open(file_str.c_str());
  
  // Generator.
  Pythia pythia;
  Event& event = pythia.event;
  
  // Random seed based on time
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + seed);

  // Display settings
  pythia.readString("Next:numberCount = 10000");
  pythia.readString("Next:numberShowEvent = 1");
  pythia.readString("Next:numberShowProcess = 1");
  
  // Collider settings
  //pythia.readString("Beams:idA =  11");
  //pythia.readString("Beams:idB = -11");
  //double mZ = pythia.particleData.m0(23);
  //pythia.settings.parm("Beams:eCM", mZ);
  pythia.readString("Beams:eCM = 13000.");
  //pythia.readString("PDF:lepton = off");
  
  // Process selection
  //pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  //pythia.readString("23:onMode = off");
  //pythia.readString("23:onIfAny = 1 2 3 4 5");
  //pythia.readString("HardQCD:all = on");
  //pythia.readString("SoftQCD:all = on");
  //pythia.readString("SoftQCD:elastic = on");
  pythia.readString("HardQCD:qq2qq = on");
  pythia.readString("PhaseSpace:pTHatMin = 500.");
  
  pythia.init();
  
  // Begin event loop. Generate event. Skip if error. 
  int n_u_charges = 0;
  int n_d_charges = 0;
  int n_events = 0;
  while (n_d_charges < NJETS) {

    // skipping if there's an error
    if (!pythia.next()) continue;

    // This object will store the recursive daughter lists of all uu->uu and dd->dd
    // processes. Once we have clustered jets, then we will search this list for
    // which jets belong to which processes.
    EventDL edl = EventDL(); 

    // This will contain final state observable particles.
    vector<PseudoJet> particles;

    // This loop:
    // -checks to see if the particle is u or d
    // -if so, checks if its mothers have different indices
    // -if so, checks if the mothers have exactly two daughters
    // -if so, checks if all 4 particles are either u or d
    // -given all this, generate a recursive daughter list for the mothers
    // -filter by final state particles and add to edl (above)
    for (unsigned int i = 0; i < event.size(); ++i) {
      int id = event[i].id();
      if (id == 1 || id == 2) {
        int m1 = event[i].mother1();
        int m2 = event[i].mother2();
        vector<int> m1dl = event[m1].daughterList();
        if (m1 != m2 && m1dl.size() == 2) {
          int id1 = event[m1].id();
          int id2 = event[m2].id();
          if ((id1 == id2) && (id1 == event[m1dl[0]].id()) && (id1 == event[m1dl[1]].id())) {
            vector<int> dl = event[m1].daughterListRecursive();
            vector<int> fsdl;
            for (int z = 0; z < dl.size(); ++z)
              if (event[dl[z]].isFinal())
                fsdl.push_back(dl[z]);
            edl.add(m1, m2, fsdl);
          }
        }
      }

      if (!event[i].isFinal()) continue;

      // no neutrinos
      if (abs(id) == 12 || abs(id) == 14 || abs(id)== 16) continue;
      
      // give remaining particle to fastjet, store id and other info
      PseudoJet particle(event[i].px(), event[i].py(), event[i].pz(), event[i].e());
      particle.set_user_info(new ID(id, event[i].charge(), event[i].col(), event[i].acol(), event[i].pT(), i, event[i].eta(), event[i].phi()));
      particles.push_back(particle);

    }  // end loop (particles)

    // Cluster final-state particles using anti kt
    JetDefinition jet_def(antikt_algorithm, 0.4);
    ClusterSequence clust_seq(particles, jet_def);
    vector<PseudoJet> jets = clust_seq.inclusive_jets();
        
    // taking the hardest uu and dd jets. also, if there are both kinds of
    // particle in a jet, putting them into ud
    PseudoJet uuj = PseudoJet(0, 0, 0, 0);
    PseudoJet ddj = PseudoJet(0, 0, 0, 0);
    
    // This loop ignores jets with |\eta| > 2.5 and checks which jet
    // constituents are parts of a uu->uu or dd->dd process. 
    // If constituents are part of one of these processes and the jet is harder
    // than the last found jet, the jet gets stored.  
    // If constituents belong to both processes, the jet gets stored in a
    // separate container
    for (unsigned int j = 0; j < jets.size(); ++j) {

      // ignoring jets with |\eta| > 2.5
      if (abs(jets[j].pseudorapidity()) > 2.5) continue;

      vector<PseudoJet> constituents = jets[j].constituents();
      bool u = false;
      bool d = false;
      
      // this loop checks if any constituents belong to a
      // process of interest
      for (unsigned int i = 0; i < constituents.size(); ++i) {
        vector<int> mothers = edl.find(constituents[i].user_info<ID>().get_index(), event);
        if (!mothers.empty()) {
          if (event[mothers[0]].id() == 1) {
            d = true;
          } else {
            u = true;
          }
        }
      } // end loop (jet constituents)

      // if this jet contains daughters from both u and d processes, discard it
      // else if this jet is harder than any before it, store it
      if (u && d)
        continue;
      else if (u && (jets[j].e() > uuj.e()))
        uuj = jets[j];
      else if (d && (jets[j].e() > ddj.e()))
        ddj = jets[j];

    } // end loop (jets)

    ++n_events;

    // if there's no jet charge to compute, continue
    if (ddj.e() < 0.1 && uuj.e() < 0.1)
      continue;

    // print the hardest jets
    if (ddj.e() > 0.1) {
      vector<PseudoJet> constituents = ddj.constituents();
      file << "D " << n_d_charges << " " << ddj.eta() << " " << ddj.phi_std() << " " << ddj.pt() << "\n";
      for (int z = 0; z < constituents.size(); ++z) {
        ID info = constituents[z].user_info<ID>();
        if (info.get_color() == 0 && info.get_acolor() == 0) {
          file << info.get_eta() << " " << info.get_phi() << " " <<  info.get_charge() << " " << info.get_pt() << "\n";
        }
      }
      ++n_d_charges;
    }
    if (uuj.e() > 0.1) {
      vector<PseudoJet> constituents = uuj.constituents();
      file << "U " << n_u_charges << " " << uuj.eta() << " " << uuj.phi_std() << " " << uuj.pt() << "\n";
      for (int z = 0; z < constituents.size(); ++z) {
        ID info = constituents[z].user_info<ID>();
        if (info.get_color() == 0 && info.get_acolor() == 0) {
          file << info.get_eta() << " " << info.get_phi() << " " <<  info.get_charge() << " " << info.get_pt() << "\n";
        }
      }
      ++n_u_charges;
    }
  } // end loop (events)
  
  pythia.stat();
  file.close();
  return 0;
}
