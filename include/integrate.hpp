#ifndef _LID_INTEGRATE_HPP
#define _LID_INTEGRATE_HPP

namespace LID {
	struct detector {
		real_t x0[3], nhat[3];
		
		detector() {}
		detector(real_t x0[3], real_t nhat[3]) {
			for (int i = 0; i < 3; i++) {
				this->x0[i] = x0[i];
				this->nhat[i] = nhat[i];
			}
		}
	};

	real_t *line_integrated_density(struct dream_data*, struct detector*);
}

#endif/*_LID_INTEGRATE_HPP*/
