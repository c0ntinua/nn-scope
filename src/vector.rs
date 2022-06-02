#![allow(dead_code)]
#![allow(unused_imports)]
use rand::random;
//use rand::*;
use num::signum;
use rand::prelude::*;
//use rand_distr::StandardNormal;
use rand_distr::{Distribution, Normal, NormalError};
//use rand::distributions::{StandardNormal, Distribution};

use rand::thread_rng;


pub fn clean( m : f64, v : &mut Vec<f64>, ) {	   	
	for x in v {
    	if x.abs() > m {
    		*x = m*signum(*x);
	    }
    }
}

pub fn random_vector(n : usize) -> Vec<f64> {
	let mut y = vec![0.0;n];
	for v in &mut y {
		*v = 1.0 - 2.0*rand::random::<f64>();
	}
	assert!(y.len() == n);	
	y	
}

pub fn random_in( r : (f64,f64) ) -> f64{
	debug_assert!(r.0 < r.1);
	r.0 + (r.1-r.0)*rand::random::<f64>()
}


pub fn random_vector_in(n: usize, r : (f64,f64)) -> Vec<f64> {
	let mut y = vec![0.0;n];
	for v in &mut y {
		*v = r.0 + (r.1-r.0)*rand::random::<f64>();
	}
	y
}
	

pub fn random_normal()  -> f64 {
	let mut sum = 0.0;
	let n = 10;
	for _ in 0..n {
		sum += random_in((-1.0,1.0));
	}
	sum / (n as f64)
	
}

	

pub fn report(v : &[f64], p : usize) {
	for x in v { print!("{:+.p$}  " ,x, p=p); }
	print!("\n");	
}

pub fn difference(x : &[f64], y: &[f64]) -> Vec<f64> {
	assert!(x.len() == y.len());
	let mut z : Vec<f64> = Vec::with_capacity(x.len());
	for i in 0..x.len() {
		z.push(x[i]-y[i]);	
	}
	z
}

pub fn sum(x : &[f64], y: &[f64]) -> Vec<f64> {
	assert!(x.len() == y.len());
	let mut z : Vec<f64> = Vec::with_capacity(x.len());
	for i in 0..x.len() {
		z.push(x[i]+y[i]);	
	}
	z
}

pub fn abs_sum(x : &[f64], y: &[f64]) -> Vec<f64> {
	assert!(x.len() == y.len());
	let mut z : Vec<f64> = Vec::with_capacity(x.len());
	for i in 0..x.len() {
		z.push(x[i].abs()+y[i].abs());	
	}
	z
}

pub fn scale( a : f64, x : &[f64] ) -> Vec<f64> {
	let mut z : Vec<f64> = Vec::with_capacity(x.len());
	for i in 0..x.len() {
		z.push(x[i]*a);
	}
	z
}