
#include <iostream>
#include <softlib/SFile.h>
#include "config.hpp"
#include "dreamoutput.hpp"
#include "LIDException.hpp"


using namespace LID;


/**
 * Load data from the named DREAM output file.
 */
struct dream_data *LID::load_dream_output(const char *filename) {
	SFile *sf = SFile::Create(filename, SFILE_MODE_READ);

	struct dream_data *dd = new struct dream_data;

	// Time grid
	sfilesize_t nt, nr, nr_f;
	dd->t = sf->GetList("/grid/t", &nt);
	dd->nt = nt;

	// Radial grid
	dd->r = sf->GetList("/grid/r", &nr);
	dd->r_f = sf->GetList("/grid/r_f", &nr_f);
	dd->dr = sf->GetList("/grid/dr", &nr);
	dd->R0 = sf->GetScalar("/grid/R0");
	dd->nr = nr;

	// Flux surfaces
	sfilesize_t eqdims[2], eqdims_f[2], ndim;
	dd->ROverR0 = sf->GetMultiArray_linear("/grid/eq/ROverR0", 2, ndim, eqdims);
	dd->ROverR0_f = sf->GetMultiArray_linear("/grid/eq/ROverR0_f", 2, ndim, eqdims_f);
	dd->Z = sf->GetMultiArray_linear("/grid/eq/Z", 2, ndim, eqdims);
	dd->Z_f = sf->GetMultiArray_linear("/grid/eq/Z_f", 2, ndim, eqdims_f);

	sfilesize_t ntheta;
	sf->GetList("/grid/eq/theta", &ntheta);
	dd->ntheta = ntheta;

	// Electron density
	sfilesize_t dims[2];
	dd->ne = sf->GetMultiArray_linear("/eqsys/n_cold", 2, ndim, dims);

	if (dims[0] != nt || dims[1] != nr) {
		throw LIDException(
			"Expected cold electron density to have dimensions "
			"%zu x %zu. Found dimensions %zu x %zu.",
			nt, nr, dims[0], dims[1]
		);
	}

	sf->Close();

	return dd;
}


