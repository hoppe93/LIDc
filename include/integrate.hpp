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

	real_t *line_integrated_density(struct dream_data*, struct detector*, real_t *L=nullptr);
	real_t line_integrated_density_at_time(len_t, struct dream_data*, struct detector*, real_t *L=nullptr);
	real_t *greens_function(struct dream_data*, const len_t, real_t**, real_t**);
	len_t find_time(real_t, struct dream_data*);
}

#endif/*_LID_INTEGRATE_HPP*/
