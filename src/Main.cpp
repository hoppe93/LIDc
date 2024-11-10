/**
 * Line integrated density computation.
 */

#include <iostream>
#include <softlib/SFile.h>
#include "config.hpp"
#include "dreamoutput.hpp"
#include "integrate.hpp"
#include "LIDException.hpp"

using namespace std;
using namespace LID;


/**
 * Program entry point.
 */
int main(int argc, char *argv[]) {
	if (argc != 2) {
		cerr << "ERROR: Expected exactly one command-line argument: name of DREAM output file missing." << endl;
		return 1;
	}

	real_t x0[3] = {0.9030, 0, 0.65};
	real_t nhat[3] = {0, 0, -1};
	struct detector *det = new struct detector(x0, nhat);

	int exit_code = 0;
	try {
		struct dream_data *dd = load_dream_output(argv[1]);
		cout << "Loaded " << dd->nt << " time steps." << endl;

		real_t *n = line_integrated_density(dd, det);

		SFile *sf = SFile::Create("line-integrated-density.h5", SFILE_MODE_WRITE);
		sf->WriteList("ne", n, dd->nt);
		sf->WriteList("t", dd->t, dd->nt);
		sf->Close();

		delete dd;
	} catch (LIDException &ex) {
		cerr << "ERROR: " << ex.what() << endl;
		exit_code = 2;
	}

	return exit_code;
}

