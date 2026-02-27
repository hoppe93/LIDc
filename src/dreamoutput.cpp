
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
	dd->Z0 = sf->GetScalar("/grid/eq/Z0");
	dd->nr = nr;

	// Flux surfaces
	sfilesize_t eqdims[2], eqdims_f[2], ndim;
	dd->RMinusR0 = sf->GetMultiArray_linear("/grid/eq/RMinusR0", 2, ndim, eqdims);
	dd->RMinusR0_f = sf->GetMultiArray_linear("/grid/eq/RMinusR0_f", 2, ndim, eqdims_f);
	dd->ZMinusZ0 = sf->GetMultiArray_linear("/grid/eq/ZMinusZ0", 2, ndim, eqdims);
	dd->ZMinusZ0_f = sf->GetMultiArray_linear("/grid/eq/ZMinusZ0_f", 2, ndim, eqdims_f);

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


struct dream_data *LID::load_luke_equilibrium(const char *filename) {
	SFile *sf = SFile::Create(filename, SFILE_MODE_READ);

	struct dream_data *dd = new struct dream_data;
	dd->nt = 0;

	sfilesize_t eqdims[2], ndim;
	dd->R0 = sf->GetScalar("/equil/Rp");
	dd->Z0 = sf->GetScalar("/equil/Zp");

	dd->RMinusR0_f = sf->GetMultiArray_linear("/equil/ptx", 2, ndim, eqdims);
	dd->ZMinusZ0_f = sf->GetMultiArray_linear("/equil/pty", 2, ndim, eqdims);

	dd->nr = eqdims[1]-1;
	dd->ntheta = eqdims[0];
	dd->r_f = new real_t[dd->nr+1];
	dd->dr = new real_t[dd->nr];
	for (len_t i = 0; i < dd->nr+1; i++)
		dd->r_f[i] = dd->RMinusR0_f[i];
	
	// Calculate distribution grid
	dd->RMinusR0 = new real_t[dd->ntheta*dd->nr];
	dd->ZMinusZ0 = new real_t[dd->ntheta*dd->nr];
	for (len_t ir = 0; ir < dd->nr; ir++) {
		for (len_t it = 0; it < dd->ntheta; it++) {
			dd->RMinusR0[it*dd->nr + ir] = 0.5*(
				dd->RMinusR0_f[it*(dd->nr+1) + (ir+1)] +
				dd->RMinusR0_f[it*(dd->nr+1) + ir]
			);
			dd->ZMinusZ0[it*dd->nr + ir] = 0.5*(
				dd->ZMinusZ0_f[it*(dd->nr+1) + (ir+1)] +
				dd->ZMinusZ0_f[it*(dd->nr+1) + ir]
			);
		}
	}
	
	dd->ne = new real_t[dd->nr];
	dd->r = new real_t[dd->nr];

	for (len_t i = 0; i < dd->nr; i++) {
		dd->r[i] = 0.5*(dd->r_f[i+1] + dd->r_f[i]);
		dd->dr[i] = dd->r_f[i+1] - dd->r_f[i];
		dd->ne[i] = 0;
	}

	return dd;
}

