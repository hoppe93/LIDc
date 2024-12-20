#ifndef _LID_DREAM_OUTPUT_HPP
#define _LID_DREAM_OUTPUT_HPP

#include "config.hpp"


namespace LID {
	struct dream_data {
		// Time and radial grid
		len_t nt, nr, ntheta;
		real_t *t, *r, *r_f;
		real_t *dr;
		real_t R0, Z0;
		// Flux surfaces (ntheta x nr)
		real_t *RMinusR0, *RMinusR0_f;
		real_t *ZMinusZ0, *ZMinusZ0_f;
		// Electron density (nt x nr)
		real_t *ne;
	};

	struct dream_data *load_dream_output(const char *filename);
}

#endif/*_LID_DREAM_OUTPUT_HPP*/
