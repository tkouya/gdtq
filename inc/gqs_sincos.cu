#ifndef __GQS_SIN_COS_CU__
#define __GQS_SIN_COS_CU__

//#include "common.cuh"
//#include "gqd_sincos.cuh"
#include "gqs.cuh"

/*
 * Note (CUDA 13 port):
 * The original source wrapped this entire file in `#ifdef USE_GQD_SIN`,
 * which caused the `sin(const gqs_real&)`, `cos(const gqs_real&)`,
 * `tan(const gqs_real&)` etc. symbols to be missing from the device
 * object whenever the macro was not defined at compile time. However
 * the matching header `gqd_sincos.cuh` declares those functions
 * unconditionally, so callers would link/ptxas-fail with
 *     Unresolved extern function '_Z3sinRK7double4'
 * on anything that used them (e.g. the test kernels in gqdtest_kernel.cu).
 * Since the header is unconditional, the definitions must be too --
 * the guard has been removed for consistency.
 */

__device__
void sincos_taylor(const gqs_real &a, 
				   gqs_real &sin_a, gqs_real &cos_a) 
{
	const float thresh = 0.5 * _qs_eps * fabs(to_float(a));
	gqs_real p, s, t, x;

	if (is_zero(a)) {
		sin_a.x = sin_a.y = sin_a.z = sin_a.w = 0.0;
		cos_a.x = 1.0;
		cos_a.y = cos_a.z = cos_a.w = 0.0;
		return;
	}

	//x = -sqr(a);
	x = negative( sqr(a) );
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * qs_inv_fact[i];
		s = s + t;
		i = i + 2;
	} while (i < n_qs_inv_fact && fabs(to_float(t)) > thresh);

	sin_a = s;
	cos_a = sqrt(1.0 - sqr(s));
}


__device__
gqs_real sin_taylor(const gqs_real &a) {
	const float thresh = 0.5 * _qs_eps * fabs(to_float(a));
	gqs_real p, s, t, x;

	if (is_zero(a)) {
		//return make_qs(0.0);
		s.x = s.y = s.z = s.w = 0.0;
		return s;
	}

	//x = -sqr(a);
	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * qs_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_qs_inv_fact && fabs(to_float(t)) > thresh);

	return s;
}


__device__
gqs_real cos_taylor(const gqs_real &a) {
	const float thresh = 0.5 * _qs_eps;
	gqs_real p, s, t, x;

	if (is_zero(a)) {
		//return make_qs(1.0);
		s.x = 1.0;
		s.y = s.z = s.w = 0.0;
		return s;
	}

	//x = -sqr(a);
	x = negative(sqr(a));
	s = 1.0 + mul_pwr2(x, 0.5);
	p = x;
	int i = 1;
	do {
		p = p * x;
		t = p * qs_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_qs_inv_fact && fabs(to_float(t)) > thresh);

	return s;
}


__device__
gqs_real sin(const gqs_real &a) {

	gqs_real z, r;
	if (is_zero(a)) {
		//return make_qs(0.0);
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	// approximately reduce modulo 2*pi
	z = nint(a / _qs_2pi);
	r = a - _qs_2pi * z;

	// approximately reduce modulo pi/2 and then modulo pi/1024
	float q = floor(r.x / _qs_pi2.x + 0.5);
	gqs_real t = r - _qs_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qs_pi1024.x + 0.5);
	t = t - _qs_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//gqs_real::error("(gqs_real::sin): Cannot reduce modulo pi/2.");
		//return gqs_real::_nan;
		//return make_qs(0.0);
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	if (abs_k > 256) {
		//gqs_real::error("(gqs_real::sin): Cannot reduce modulo pi/1024.");
		//return gqs_real::_nan;
		//return make_qs( 0.0 );
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	if (k == 0) {
		switch (j) {
	  case 0:
		  return sin_taylor(t);
	  case 1:
		  return cos_taylor(t);
	  case -1:
		  return negative(cos_taylor(t));
	  default:
		  return negative(sin_taylor(t));
		}
	}

	//gqs_real sin_t, cos_t;
	//gqs_real u = d_qs_cos_table[abs_k-1];
	//gqs_real v = d_qs_sin_table[abs_k-1]; 
	//sincos_taylor(t, sin_t, cos_t);
	///use z and r again to avoid allocate additional memory
	///z = sin_t, r = cos_t
	sincos_taylor( t, z, r );

	if (j == 0) {
		z = d_qs_cos_table[abs_k-1] * z;
		r = d_qs_sin_table[abs_k-1] * r;
		if (k > 0) {
			//z = d_qs_cos_table[abs_k-1] * z;
			//r = d_qs_sin_table[abs_k-1] * r;
			return  z + r;
		} else {
			//z = d_qs_cos_table[abs_k-1] * z;
			//r = d_qs_sin_table[abs_k-1] * r;
			return z - r;
		}
	} else if (j == 1) {
		r = d_qs_cos_table[abs_k-1] * r;
		z = d_qs_sin_table[abs_k-1] * z;
		if (k > 0) {
			//r = d_qs_cos_table[abs_k-1] * r;
			//z = d_qs_sin_table[abs_k-1] * z;
			return r - z;
		} else {
			//r = d_qs_cos_table[abs_k-1] * r;
			//z = d_qs_sin_table[abs_k-1] * z;
			return r + z;
		}
	} else if (j == -1) {
		z = d_qs_sin_table[abs_k-1] * z;
		r = d_qs_cos_table[abs_k-1] * r;
		if (k > 0) {
			//z = d_qs_sin_table[abs_k-1] * z;
			//r = d_qs_cos_table[abs_k-1] * r;
			return z - r;
		} else {
			//r = negative(d_qs_cos_table[abs_k-1]) * r;
			//r = (d_qs_cos_table[abs_k-1]) * r;
			r.x = -r.x;
			r.y = -r.y;
			r.z = -r.z;
			r.w = -r.w;
			//z = d_qs_sin_table[abs_k-1] * z;
			return r - z;
		}
	} else {
		r = d_qs_sin_table[abs_k-1] * r ;
		z = d_qs_cos_table[abs_k-1] * z;
		if (k > 0) {
			//z = negative(d_qs_cos_table[abs_k-1]) * z;
			//z = d_qs_cos_table[abs_k-1] * z;
			z.x = -z.x;
			z.y = -z.y;
			z.z = -z.z;
			z.w = -z.w;
			//r = d_qs_sin_table[abs_k-1] * r;
			return z - r;
		} else {
			//r = d_qs_sin_table[abs_k-1] * r ;
			//z = d_qs_cos_table[abs_k-1] * z;
			return r - z;
		}
	}
}


__device__
gqs_real cos(const gqs_real &a) {
	if (is_zero(a)) {
		return make_qs(1.0);
	}

	// approximately reduce modulo 2*pi
	gqs_real z = nint(a / _qs_2pi);
	gqs_real r = a - _qs_2pi * z;

	// approximately reduce modulo pi/2 and then modulo pi/1024
	float q = floor(r.x / _qs_pi2.x + 0.5);
	gqs_real t = r - _qs_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qs_pi1024.x + 0.5);
	t = t - _qs_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//qd_real::error("(qd_real::cos): Cannot reduce modulo pi/2.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	if (abs_k > 256) {
		//qd_real::error("(qd_real::cos): Cannot reduce modulo pi/1024.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	if (k == 0) {
		switch (j) {
	  		case 0:
		  		return cos_taylor(t);
			  case 1:
				  return negative(sin_taylor(t));
			  case -1:
				  return sin_taylor(t);
			  default:
				  return negative(cos_taylor(t));
		}
	}

	gqs_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gqs_real u = d_qs_cos_table[abs_k - 1];
	gqs_real v = d_qs_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) {
			r = u * cos_t - v * sin_t;
		} else {
			r = u * cos_t + v * sin_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			r = negative(u * sin_t) - v * cos_t;
		} else {
			r = v * cos_t - u * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			r = u * sin_t + v * cos_t;
		} else {
			r = u * sin_t - v * cos_t;
		}
	} else {
		if (k > 0) {
			r = v * sin_t - u * cos_t;
		} else {
			r = negative(u * cos_t) - v * sin_t;
		}
	}

	return r;
}


__device__
void sincos(const gqs_real &a, gqs_real &sin_a, gqs_real &cos_a) {

	if (is_zero(a)) {
		sin_a = make_qs(0.0);
		cos_a = make_qs(1.0);
		return;
	}

	// approximately reduce by 2*pi
	gqs_real z = nint(a / _qs_2pi);
	gqs_real t = a - _qs_2pi * z;

	// approximately reduce by pi/2 and then by pi/1024.
	float q = floor(t.x / _qs_pi2.x + 0.5);
	t = t - _qs_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qs_pi1024.x + 0.5);
	t = t - _qs_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//qd_real::error("(qd_real::sincos): Cannot reduce modulo pi/2.");
		//cos_a = sin_a = qd_real::_nan;
		cos_a = sin_a = make_qs(0.0);
		return;
	}

	if (abs_k > 256) {
		//qd_real::error("(qd_real::sincos): Cannot reduce modulo pi/1024.");
		//cos_a = sin_a = qd_real::_nan;
		cos_a = sin_a = make_qs(0.0);
		return;
	}

	gqs_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if (j == 0) {
			sin_a = sin_t;
			cos_a = cos_t;
		} else if (j == 1) {
			sin_a = cos_t;
			cos_a = negative(sin_t);
		} else if (j == -1) {
			sin_a = negative(cos_t);
			cos_a = sin_t;
		} else {
			sin_a = negative(sin_t);
			cos_a = negative(cos_t);
		}
		return;
	}

	gqs_real u = d_qs_cos_table[abs_k - 1];
	gqs_real v = d_qs_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) {
			sin_a = u * sin_t + v * cos_t;
			cos_a = u * cos_t - v * sin_t;
		} else {
			sin_a = u * sin_t - v * cos_t;
			cos_a = u * cos_t + v * sin_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			cos_a = negative(u * sin_t) - v * cos_t;
			sin_a = u * cos_t - v * sin_t;
		} else {
			cos_a = v * cos_t - u * sin_t;
			sin_a = u * cos_t + v * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			cos_a = u * sin_t + v * cos_t;
			sin_a =  v * sin_t - u * cos_t;
		} else {
			cos_a = u * sin_t - v * cos_t;
			sin_a = negative(u * cos_t) - v * sin_t;
		}
	} else {
		if (k > 0) {
			sin_a = negative(u * sin_t) - v * cos_t;
			cos_a = v * sin_t - u * cos_t;
		} else {
			sin_a = v * cos_t - u * sin_t;
			cos_a = negative(u * cos_t) - v * sin_t;
		}
	}
}


__device__
gqs_real tan(const gqs_real &a) {
  gqs_real s, c;
  sincos(a, s, c);
  return s/c;
}

#ifdef ALL_MATH	

__device__
gqs_real atan2(const gqs_real &y, const gqs_real &x) {

	if (is_zero(x)) {

		if (is_zero(y)) {
			// Both x and y is zero. 
			//qd_real::error("(qd_real::atan2): Both arguments zero.");
			//return qd_real::_nan;
			return make_qs(0.0);
		}

		return (is_positive(y)) ? _qs_pi2 : negative(_qs_pi2);
	} else if (is_zero(y)) {
		return (is_positive(x)) ? make_qs(0.0) : _qs_pi;
	}

	if (x == y) {
		return (is_positive(y)) ? _qs_pi4 : negative(_qs_3pi4);
	}

	if (x == negative(y)) {
		return (is_positive(y)) ? _qs_3pi4 : negative(_qs_pi4);
	}

	gqs_real r = sqrt(sqr(x) + sqr(y));
	gqs_real xx = x / r;
	gqs_real yy = y / r;

	gqs_real z = make_qs(atan2(to_float(y), to_float(x)));
	gqs_real sin_z, cos_z;

	if (abs(xx.x) > abs(yy.x)) {
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
	} else {
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
	}

	return z;
}


__device__
gqs_real atan(const gqs_real &a) {
	return atan2(a, make_qs(1.0));
}


__device__
gqs_real asin(const gqs_real &a) {
	gqs_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//qd_real::error("(qd_real::asin): Argument out of domain.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? _qs_pi2 : negative(_qs_pi2);
	}

	return atan2(a, sqrt(1.0 - sqr(a)));
}


__device__
gqs_real acos(const gqs_real &a) {
	gqs_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//qd_real::error("(qd_real::acos): Argument out of domain.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? make_qs(0.0) : _qs_pi;
	}

	return atan2(sqrt(1.0 - sqr(a)), a);
}


__device__
gqs_real sinh(const gqs_real &a) {
	if (is_zero(a)) {
		return make_qs(0.0);
	}

	if (abs(a) > 0.05) {
		gqs_real ea = exp(a);
		return mul_pwr2(ea - inv(ea), 0.5);
	}

	gqs_real s = a;
	gqs_real t = a;
	gqs_real r = sqr(t);
	float m = 1.0;
	float thresh = abs(to_float(a) * _qs_eps);

	do {
		m = m + 2.0;
		t = (t*r);
		t = t/((m-1) * m);

		s = s + t;
	} while (abs(t) > thresh);

	return s;
}


__device__
gqs_real cosh(const gqs_real &a) {
	if (is_zero(a)) {
		return make_qs(1.0);
	}

	gqs_real ea = exp(a);
	return mul_pwr2(ea + inv(ea), 0.5);
}


__device__
gqs_real tanh(const gqs_real &a) {
	if (is_zero(a)) {
		return make_qs(0.0);
	}

	if (abs(to_float(a)) > 0.05) {
		gqs_real ea = exp(a);
		gqs_real inv_ea = inv(ea);
		return (ea - inv_ea) / (ea + inv_ea);
	} else {
		gqs_real s, c;
		s = sinh(a);
		c = sqrt(1.0 + sqr(s));
		return s / c;
	}
}


__device__
void sincosh(const gqs_real &a, gqs_real &s, gqs_real &c) {
	if (abs(to_float(a)) <= 0.05) {
		s = sinh(a);
		c = sqrt(1.0 + sqr(s));
	} else {
		gqs_real ea = exp(a);
		gqs_real inv_ea = inv(ea);
		s = mul_pwr2(ea - inv_ea, 0.5);
		c = mul_pwr2(ea + inv_ea, 0.5);
	}
}


__device__
gqs_real asinh(const gqs_real &a) {
	return log(a + sqrt(sqr(a) + 1.0));
}


__device__
gqs_real acosh(const gqs_real &a) {
	if (a < 1.0) {
		///qd_real::error("(qd_real::acosh): Argument out of domain.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	return log(a + sqrt(sqr(a) - 1.0));
}


__device__
gqs_real atanh(const gqs_real &a) {
	if (abs(a) >= 1.0) {
		//qd_real::error("(qd_real::atanh): Argument out of domain.");
		//return qd_real::_nan;
		return make_qs(0.0);
	}

	return mul_pwr2(log((1.0 + a) / (1.0 - a)), 0.5);
}


#endif /* ALL_MATH */

/* (The matching `#endif // USE_GQD_SIN` was removed together with the
   opening `#ifdef USE_GQD_SIN` at the top of the file.) */

#endif /* __GQS_SIN_COS_CU__ */


