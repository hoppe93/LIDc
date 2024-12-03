/**
 * Routines for carrying out line integration.
 */

#include <cmath>
#include <omp.h>
#include <iostream>
#include "config.hpp"
#include "dreamoutput.hpp"
#include "integrate.hpp"
#include "LIDException.hpp"


using namespace LID;
using namespace std;


/**
 * Check if the line-of-sight originating in x0, directed along nhat,
 * intersects the line extending between (x0, z0) and (x1, z1).
 */
namespace LID {
void los_intersection(
	real_t detX0[3], real_t nhat[3],
	const real_t x0, const real_t z0,
	const real_t x1, const real_t z1,
	real_t *l1, real_t *l2
) {
	real_t X0 = detX0[0], Y0 = detX0[1], Z0 = detX0[2];
	real_t nx = nhat[0], ny = nhat[1], nz = nhat[2];

	real_t a0 = x0*x0 - X0*X0 - Y0*Y0 + 2*x0*(Z0-z0)*(x1-x0)/(z1-z0) +
				(Z0-z0)*(Z0-z0) * (x1-x0)*(x1-x0) / ((z1-z0)*(z1-z0));
	
	real_t a1 = 2*x0*nz*(x1-x0)/(z1-z0) +
				2*nz*(Z0-z0)*(x1-x0)*(x1-x0)/((z1-z0)*(z1-z0)) -
				2*(X0*nx + Y0*ny);
	
	real_t a2 = nz*nz * (x1-x0)*(x1-x0) / ((z1-z0)*(z1-z0)) - nx*nx - ny*ny;

	// Check for real solution
	real_t sqr = -a0/a2 + a1*a1 / (4*a2*a2);
	if (sqr < 0) {
		*l1 = -1;
		*l2 = -1;
		return;
	}

	// Obtain the two roots
	real_t _l1 = -a1/(2*a2) + sqrt(sqr);
	real_t _l2 = -a1/(2*a2) - sqrt(sqr);

	// Check for 0 < t < 1
	real_t t1 = (Z0-z0 + _l1*nz)/(z1-z0);
	real_t t2 = (Z0-z0 + _l2*nz)/(z1-z0);

	if (t1 < 0 || t1 > 1)
		*l1 = -1;
	else *l1 = _l1;

	if (t2 < 0 || t2 > 1)
		*l2 = -1;
	else *l2 = _l2;
}


/**
 * Find the distances along the line-of-sight where the specified
 * flux surface is intersected.
 */
void find_intersections(
	len_t isurf, struct dream_data *dd, real_t detX0[3], real_t nhat[3],
	real_t *l1, real_t *l2
) {
	real_t *R = dd->RMinusR0_f;
	real_t *Z = dd->ZMinusZ0_f;
	real_t R0 = dd->R0;
	real_t Z0 = dd->Z0;
	const len_t nr = dd->nr;

	len_t n = 0;
	real_t l[3] = {-1, -1, -1};

	// Calculate all intersections (stop once two intersections
	// have been located; this is the maximum number of possible
	// intersections)
	for (len_t i = 0; i < dd->ntheta && n < 2; i++) {
		real_t x0 = R[i*(nr+1) + isurf] + R0;
		real_t z0 = Z[i*(nr+1) + isurf] + Z0;

		real_t x1, z1;
		if (i == dd->ntheta-1) {
			x1 = R[0*(nr+1) + isurf] + R0;
			z1 = Z[0*(nr+1) + isurf] + Z0;
		} else {
			x1 = R[(i+1)*(nr+1) + isurf] + R0;
			z1 = Z[(i+1)*(nr+1) + isurf] + Z0;
		}

		real_t _l1, _l2;
		los_intersection(detX0, nhat, x0, z0, x1, z1, &_l1, &_l2);

		if (_l1 > 0)
			l[n++] = _l1;
		if (_l2 > 0)
			l[n++] = _l2;
	}

	if (l[0] < l[1]) {
		*l1 = l[0];
		*l2 = l[1];
	} else {
		*l1 = l[1];
		*l2 = l[2];
	}
}


/**
 * Evaluate the line-integrated electron density in the given
 * time step of the specified DREAM output struct.
 */
real_t integrate(
	len_t it, struct dream_data *dd,
	real_t x0[3], real_t nhat[3], real_t *length=nullptr
) {
	const len_t nr = dd->nr;
	real_t l11, l12, l21, l22;
	find_intersections(dd->nr, dd, x0, nhat, &l11, &l12);

	// No intersection of LOS with plasma?
	if (l11 < 0)
		return 0;

	real_t n = 0;
	real_t L = 0;
	real_t minl = 10000, maxl = std::max(l11, l12);

	if (l11 >= 0 && l11 < l12)
		minl = l11;
	else if (l12 >= 0)
		minl = l12;

	for (len_t ir = dd->nr; ir > 0; ir--) {
		if (ir == 1)
			l21 = -1, l22 = -1;
		else
			find_intersections(ir-1, dd, x0, nhat, &l21, &l22);

		// Evaluate integral
		// If the LOS does not intersect the inner flux-surface,
		// this is where the LOS turns and we instead connect l11 -> l12.
		if (l21 < 0) {
			n += dd->ne[it*nr + (ir-1)] * std::abs(l11-l12);
			L += std::abs(l11-l12);
		} else {
			n += dd->ne[it*nr + (ir-1)] * (std::abs(l11-l21) + std::abs(l12-l22));
			L += std::abs(l11-l21) + std::abs(l12-l22);
		}

		l11 = l21;
		l12 = l22;

		if (l11 >= 0 && l11 < minl) minl = l11;
		if (l12 >= 0 && l12 < minl) minl = l12;

		if (l11 >= 0 && l11 > maxl) maxl = l11;
		if (l12 >= 0 && l12 > maxl) maxl = l12;
	}

	// If requested, return length of LOS
	if (length != nullptr)
		*length = L;

	return n;
}


/**
 * Calculate the line-integrated electron density for the given
 * DREAM output struct.
 */
real_t *line_integrated_density(
	struct dream_data *dd, struct detector *det,
	real_t *length
) {
	real_t *nlin = new real_t[dd->nt];
	real_t L = 0;

	#pragma omp parallel for shared(nlin,L) 
	for (len_t i = 0; i < dd->nt; i++) {
		nlin[i] = integrate(i, dd, det->x0, det->nhat, &L);
	}

	if (length != nullptr)
		*length = L;

	return nlin;
}
}

